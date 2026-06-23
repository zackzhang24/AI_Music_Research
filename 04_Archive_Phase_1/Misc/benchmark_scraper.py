#!/usr/bin/env python3
"""
benchmark_scraper.py — AI Music Detection Benchmark
Tests all 100 tracks (human_track_*.mp3 + ai_track_*.mp3) against:
  • AHA Music     https://aha-music.com/aimusicdetector
  • SubmitHub     https://www.submithub.com/ai-song-checker

Pre-run DOM audit findings
──────────────────────────
AHA Music   : Cloudflare Turnstile CAPTCHA gates the file-input element.
              The uploader div only renders after .cf-turnstile widget resolves.
              Headless Chromium fails Turnstile (bot fingerprint); non-headless
              on Apple Silicon with GPU can pass it. Script tries non-headless
              with a 35 s Turnstile wait. If the widget never resolves the run
              is marked "CAPTCHA Blocked". Daily free quota: 5 tracks.

SubmitHub   : File uploads require a logged-in account. The 2-detection free
              tier is IP-based and was exhausted during DOM inspection. Script
              attempts upload after optional login; without credentials every
              track is marked "Login Required".

Fill in SUBMITHUB_EMAIL / SUBMITHUB_PASSWORD below (or leave blank).

Memory model : one fresh Playwright browser process per track, browser.close()
               after each. gc.collect() every iteration.
Rate-limit   : keyword scan → "Rate Limited". CAPTCHA → "CAPTCHA Blocked".
               Login wall → "Login Required". Either site failing never crashes
               the other.
Output       : benchmark_results.csv  (append + resume-safe)
Screenshots  : screenshots/  (saved on Timeout or unexpected error)
"""

import os, sys, csv, gc, re, time, glob, subprocess

# ── Optional SubmitHub credentials ───────────────────────────────────────────
SUBMITHUB_EMAIL    = ""   # e.g. "you@example.com"
SUBMITHUB_PASSWORD = ""   # e.g. "hunter2"

# ── Paths & URLs ──────────────────────────────────────────────────────────────
SONGS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "songs")
RESULTS_CSV  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.csv")
SCREENSHOTS  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")

AHA_URL       = "https://aha-music.com/aimusicdetector"
SUBMITHUB_URL = "https://www.submithub.com/ai-song-checker"
SUBMITHUB_LOGIN_URL = "https://www.submithub.com/login"

CSV_COLUMNS   = ["Filename", "True_Label", "AHA_Result", "SubmitHub_Result"]

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# ── Timing ───────────────────────────────────────────────────────────────────
PAGE_LOAD_MS     = 30_000   # page navigation timeout
TURNSTILE_WAIT_S = 35       # seconds to wait for Turnstile to resolve
RESULT_WAIT_MS   = 90_000   # max wait for analysis result
BETWEEN_TRACKS_S = 4        # polite pause between tracks

# ── Keyword lists ─────────────────────────────────────────────────────────────
RATE_LIMIT_PHRASES = [
    "limit reached", "daily limit", "upgrade to", "paywall",
    "free trial ended", "you've used", "maximum free", "out of free",
    "no more free", "subscribe to", "detections remaining: 0",
]
LOGIN_PHRASES = [
    "please log in", "login to", "sign in to", "must be logged",
    "create an account", "log in be",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def true_label(filename: str) -> str:
    if filename.startswith("ai_track_"):    return "AI"
    if filename.startswith("human_track_"): return "Human"
    return "Unknown"


def scan(text: str, phrases: list) -> bool:
    t = text.lower()
    return any(p in t for p in phrases)


def extract_score(text: str):
    """Pull the most relevant result snippet from page text."""
    # Percentage (e.g. "87%", "87.3 %")
    m = re.search(r'(\d{1,3}(?:\.\d+)?)\s*%', text)
    if m:
        # Return a broader window for context
        start = max(0, m.start() - 30)
        end   = min(len(text), m.end() + 30)
        return text[start:end].replace("\n", " ").strip()
    # Explicit verdicts
    for pat in [
        r'AI[- ]?[Gg]enerated',
        r'[Hh]uman[- ]?[Mm]ade',
        r'[Nn]ot (?:AI|ai)',
        r'AI [Pp]robability[:\s]+[\d.]+',
        r'AI [Ss]core[:\s]+[\d.]+',
        r'[Ll]ikely AI',
        r'[Ll]ikely [Hh]uman',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return None


def screenshot(page, tag: str):
    os.makedirs(SCREENSHOTS, exist_ok=True)
    path = os.path.join(SCREENSHOTS, f"{tag}.png")
    try:
        page.screenshot(path=path)
    except Exception:
        pass


def dismiss_overlays(page):
    for sel in [
        "button:has-text('Accept All')", "button:has-text('Accept')",
        "button:has-text('I agree')",    "button:has-text('OK')",
        "button:has-text('Got it')",     "[class*='cookie'] button",
        "[class*='consent'] button",     "[class*='gdpr'] button",
    ]:
        try:
            btn = page.locator(sel).first
            if btn.is_visible(timeout=800):
                btn.click()
                time.sleep(0.4)
        except Exception:
            pass


# ── AHA Music ─────────────────────────────────────────────────────────────────

def test_aha(playwright_instance, file_path: str) -> str:
    """
    Upload file to AHA Music and return result string.

    Flow:
      1. Load page — Cloudflare Turnstile widget is embedded as an iframe.
      2. Wait up to TURNSTILE_WAIT_S for the hidden file input to appear
         (it only materialises once Turnstile auto-resolves).
         Uses wait_for_selector — safe against page crashes/navigation.
      3. Set file on the input → wait for analysis result → extract score.

    Cloudflare Turnstile note:
      In headless Chromium, Turnstile almost always detects automation and
      never resolves → "CAPTCHA Blocked" for every track.
      Non-headless on Apple Silicon sometimes passes; set headless=False in
      the launch call below and ensure a display is available.
    """
    browser = ctx = page = None
    try:
        browser = playwright_instance.chromium.launch(
            headless=True,   # flip to False on a machine with a display for Turnstile
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        ctx  = browser.new_context(user_agent=UA, viewport={"width": 1280, "height": 800})
        page = ctx.new_page()

        page.goto(AHA_URL, wait_until="domcontentloaded", timeout=PAGE_LOAD_MS)
        dismiss_overlays(page)

        body = page.inner_text("body")
        if scan(body, RATE_LIMIT_PHRASES):
            return "Rate Limited"

        # ── Wait for Turnstile to resolve (file input appears after it does) ─
        print(f"    Waiting up to {TURNSTILE_WAIT_S}s for Turnstile → file input…")
        try:
            page.wait_for_selector(
                "input[type='file']",
                state="attached",
                timeout=TURNSTILE_WAIT_S * 1_000,
            )
            print("    File input appeared — Turnstile resolved ✓")
        except Exception:
            screenshot(page, f"aha_turnstile_{os.path.basename(file_path)}")
            return "CAPTCHA Blocked"

        # ── Upload file ────────────────────────────────────────────────────
        try:
            page.locator("input[type='file']").set_input_files(file_path)
        except Exception:
            try:
                with page.expect_file_chooser(timeout=10_000) as fc_info:
                    page.locator(
                        "[class*='uploader']:visible, [class*='upload']:visible, "
                        "[class*='drop']:visible"
                    ).first.click(timeout=8_000)
                fc_info.value.set_files(file_path)
            except Exception as e:
                return f"Upload Failed: {str(e)[:50]}"

        # ── Wait for result ────────────────────────────────────────────────
        try:
            page.wait_for_function(
                """() => {
                    const t = document.body.innerText.toLowerCase();
                    return t.includes('%') ||
                           t.includes('ai generated') || t.includes('human made') ||
                           t.includes('not ai')        || t.includes('likely ai') ||
                           t.includes('upgrade')       || t.includes('limit') ||
                           t.includes('error');
                }""",
                timeout=RESULT_WAIT_MS,
            )
        except Exception:
            screenshot(page, f"aha_timeout_{os.path.basename(file_path)}")
            return "Timeout"

        body = page.inner_text("body")
        if scan(body, RATE_LIMIT_PHRASES):
            return "Rate Limited"

        score = extract_score(body)
        return score if score else body[:150].replace("\n", " ").strip()

    except Exception as e:
        return f"Error: {str(e)[:70]}"
    finally:
        for obj in (page, ctx, browser):
            if obj is not None:
                try: obj.close()
                except Exception: pass


# ── SubmitHub ─────────────────────────────────────────────────────────────────

_submithub_logged_in = False   # module-level flag — only login once per run


def _submithub_login(page):
    """Attempt SubmitHub login if credentials are supplied. Returns True on success."""
    global _submithub_logged_in
    if _submithub_logged_in:
        return True
    if not SUBMITHUB_EMAIL or not SUBMITHUB_PASSWORD:
        return False

    try:
        page.goto(SUBMITHUB_LOGIN_URL, wait_until="domcontentloaded", timeout=PAGE_LOAD_MS)
        page.wait_for_timeout(2000)
        dismiss_overlays(page)
        page.fill("input[type='email'], input[name='email'], input[placeholder*='email' i]",
                  SUBMITHUB_EMAIL)
        page.fill("input[type='password']", SUBMITHUB_PASSWORD)
        page.locator(
            "button[type='submit'], button:has-text('Log in'), button:has-text('Sign in')"
        ).first.click()
        page.wait_for_load_state("domcontentloaded", timeout=PAGE_LOAD_MS)
        page.wait_for_timeout(2000)

        body = page.inner_text("body")
        if SUBMITHUB_EMAIL.split("@")[0].lower() in body.lower() or \
                "log out" in body.lower() or "sign out" in body.lower():
            _submithub_logged_in = True
            print("    SubmitHub login ✓")
            return True
        print(f"    SubmitHub login may have failed (no email in body)")
        return False
    except Exception as e:
        print(f"    SubmitHub login error: {e}")
        return False


def test_submithub(playwright_instance, file_path: str) -> str:
    """
    Upload file to SubmitHub AI Song Checker and return result string.

    Blockers found during DOM audit:
      • Free file-upload detections are IP-based (0/2 exhausted; does NOT
        reset per browser session).
      • File uploads require a logged-in SubmitHub account.
      • If SUBMITHUB_EMAIL + SUBMITHUB_PASSWORD are set above, the script
        logs in once and reuses the session cookie per context.
    """
    browser = ctx = page = None
    try:
        browser = playwright_instance.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        ctx  = browser.new_context(user_agent=UA, viewport={"width": 1280, "height": 800})
        page = ctx.new_page()

        # Login if credentials provided (only actually GETs login page first time)
        if SUBMITHUB_EMAIL and SUBMITHUB_PASSWORD and not _submithub_logged_in:
            _submithub_login(page)

        page.goto(SUBMITHUB_URL, wait_until="domcontentloaded", timeout=PAGE_LOAD_MS)
        page.wait_for_timeout(3000)
        dismiss_overlays(page)

        body = page.inner_text("body")

        if scan(body, RATE_LIMIT_PHRASES):
            return "Rate Limited"

        if not SUBMITHUB_EMAIL:
            # No credentials: detect that upload will require login
            if "free detections: 0" in body.lower() or scan(body, LOGIN_PHRASES):
                return "Login Required"

        # ── Upload file ────────────────────────────────────────────────────
        # The hidden file input is triggered by clicking #upload-box
        upload_box = page.locator("#upload-box")
        if not upload_box.count():
            return "Upload Area Missing"

        try:
            with page.expect_file_chooser(timeout=10_000) as fc_info:
                upload_box.click()
            fc_info.value.set_files(file_path)
        except Exception:
            # Fallback: set on the hidden input directly
            try:
                page.locator("input[type='file']").set_input_files(file_path)
            except Exception as e:
                return f"Upload Failed: {str(e)[:50]}"

        # ── Wait for result ────────────────────────────────────────────────
        try:
            page.wait_for_function(
                """() => {
                    const t = document.body.innerText.toLowerCase();
                    return t.includes('%') || t.includes('score') ||
                           t.includes('ai generated') || t.includes('human') ||
                           t.includes('not ai') || t.includes('log in') ||
                           t.includes('login') || t.includes('upgrade') ||
                           t.includes('limit') || t.includes('error');
                }""",
                timeout=RESULT_WAIT_MS,
            )
        except Exception:
            screenshot(page, f"submithub_timeout_{os.path.basename(file_path)}")
            return "Timeout"

        body = page.inner_text("body")

        if scan(body, LOGIN_PHRASES):
            return "Login Required"
        if scan(body, RATE_LIMIT_PHRASES):
            return "Rate Limited"

        score = extract_score(body)
        return score if score else body[:150].replace("\n", " ").strip()

    except Exception as e:
        return f"Error: {str(e)[:70]}"
    finally:
        for obj in (page, ctx, browser):
            if obj is not None:
                try: obj.close()
                except Exception: pass


# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_done() -> set:
    done = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                done.add(row.get("Filename", ""))
    return done


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright", "-q"])
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        from playwright.sync_api import sync_playwright

    files = sorted(glob.glob(os.path.join(SONGS_DIR, "*.mp3")))
    if not files:
        print(f"No MP3 files in {SONGS_DIR}")
        return

    done        = load_done()
    todo        = [f for f in files if os.path.basename(f) not in done]
    total       = len(files)
    write_header = not os.path.exists(RESULTS_CSV)

    csv_fh = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()

    print("=" * 70)
    print(f"AI Music Benchmark  —  {len(todo)} remaining of {total} tracks")
    print("=" * 70)
    print(f"  AHA Music  : {AHA_URL}")
    print(f"  SubmitHub  : {SUBMITHUB_URL}")
    print(f"  CSV output : {RESULTS_CSV}")
    if SUBMITHUB_EMAIL:
        print(f"  SubmitHub account: {SUBMITHUB_EMAIL}")
    else:
        print("  SubmitHub account: none — file uploads will return 'Login Required'")
    print()

    aha_hard_blocked = False   # once Turnstile fully fails, skip for remainder

    for idx, fpath in enumerate(todo, 1):
        fname = os.path.basename(fpath)
        label = true_label(fname)
        global_idx = len(done) + idx
        print(f"[{global_idx:03d}/{total}] {fname}  ({label})")

        # ── Isolated browser process per track, per site ──────────────────
        # Each test function spawns and closes its own browser so a crash
        # in one site's test cannot poison the other site's test.
        with sync_playwright() as p:

            # ── AHA Music ────────────────────────────────────────────────
            if aha_hard_blocked:
                aha_result = "CAPTCHA Blocked"
            else:
                aha_result = test_aha(p, fpath)
                if aha_result == "Rate Limited":
                    aha_hard_blocked = True
                    print(f"  AHA     : Rate Limited (daily cap — flagging remaining)")
            print(f"  AHA     : {aha_result}")

            # ── SubmitHub ─────────────────────────────────────────────────
            sub_result = test_submithub(p, fpath)
            print(f"  SubHub  : {sub_result}")

        # ── Write row ─────────────────────────────────────────────────────
        writer.writerow({
            "Filename":         fname,
            "True_Label":       label,
            "AHA_Result":       aha_result,
            "SubmitHub_Result": sub_result,
        })
        csv_fh.flush()

        # ── Memory cleanup ────────────────────────────────────────────────
        gc.collect()
        print(f"  ✓ Written + gc.collect()\n")

        time.sleep(BETWEEN_TRACKS_S)

    csv_fh.close()

    # ── Final summary ─────────────────────────────────────────────────────
    rows = []
    with open(RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    aha_blocked  = sum(1 for r in rows if "Blocked"   in r["AHA_Result"])
    aha_limited  = sum(1 for r in rows if "Rate"      in r["AHA_Result"])
    aha_ok       = sum(1 for r in rows if "%" in r["AHA_Result"] or
                       any(v in r["AHA_Result"] for v in ("AI", "Human", "Not")))
    sub_login    = sum(1 for r in rows if "Login"     in r["SubmitHub_Result"])
    sub_limited  = sum(1 for r in rows if "Rate"      in r["SubmitHub_Result"])
    sub_ok       = sum(1 for r in rows if "%" in r["SubmitHub_Result"] or
                       any(v in r["SubmitHub_Result"] for v in ("AI", "Human", "Not")))

    print("=" * 70)
    print(f"BENCHMARK COMPLETE  —  {len(rows)} rows in {RESULTS_CSV}")
    print("=" * 70)
    print(f"  AHA Music  : {aha_ok} scored | {aha_limited} rate-limited | {aha_blocked} CAPTCHA blocked")
    print(f"  SubmitHub  : {sub_ok} scored | {sub_limited} rate-limited | {sub_login} login required")
    print()
    if aha_blocked > 0:
        print("  ► AHA Music CAPTCHA: run with a logged-in Chrome profile or")
        print("    a CAPTCHA-solving service (2captcha / anti-captcha) to unlock.")
    if sub_login > 0:
        print("  ► SubmitHub login: set SUBMITHUB_EMAIL + SUBMITHUB_PASSWORD at")
        print("    the top of this script, then re-run (resume-safe, skips done rows).")


if __name__ == "__main__":
    main()
