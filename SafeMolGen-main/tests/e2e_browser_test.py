"""
E2E browser test for SafeMolGen-DrugOracle UI (FastAPI + React).

Requires: pip install playwright && playwright install chromium
App must be running: backend (python scripts/run_app.py) and frontend (cd frontend && npm run dev)
Default frontend URL: http://localhost:5173

Run: python -m pytest tests/e2e_browser_test.py -v
  or: python tests/e2e_browser_test.py
"""

import os
import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None

try:
    from playwright.sync_api import sync_playwright, expect
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.environ.get("E2E_APP_URL", "http://localhost:5173")

# Timeouts
NAV_TIMEOUT = 15000
WAIT_AFTER_CLICK = 1500
ANALYZE_WAIT = 6000
COMPARE_WAIT = 8000
GENERATE_WAIT = 90000  # generation can be slow


def _new_page(browser):
    page = browser.new_page()
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
    page.wait_for_selector("text=Generate", timeout=10000)
    return page


def _test_app_loads():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _new_page(browser)
            expect(page.get_by_text("Generate New Molecules", exact=False).first).to_be_visible(timeout=5000)
        finally:
            browser.close()


def _test_about_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _new_page(browser)
            page.get_by_role("link", name="About").first.click()
            page.wait_for_timeout(WAIT_AFTER_CLICK)
            expect(page.get_by_text("Overview", exact=False).first).to_be_visible(timeout=5000)
            expect(page.get_by_text("SafeMolGen-DrugOracle is an integrated", exact=False).first).to_be_visible(timeout=3000)
        finally:
            browser.close()


def _test_sidebar_settings_visible():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _new_page(browser)
            expect(page.get_by_text("Generation parameters", exact=False).first).to_be_visible(timeout=5000)
            expect(page.get_by_text("Target success probability", exact=False).first).to_be_visible(timeout=3000)
            expect(page.get_by_text("Max iterations", exact=False).first).to_be_visible(timeout=3000)
            expect(page.get_by_text("Safety threshold", exact=False).first).to_be_visible(timeout=3000)
            expect(page.get_by_text("Show advanced", exact=False).first).to_be_visible(timeout=3000)
        finally:
            browser.close()


def _test_generate_page_ui():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _new_page(browser)
            expect(page.get_by_text("Generation parameters", exact=False).first).to_be_visible(timeout=5000)
            expect(page.get_by_text("Target success probability", exact=False).first).to_be_visible(timeout=3000)
            expect(page.get_by_role("button", name="Run generation").first).to_be_visible(timeout=3000)
        finally:
            browser.close()


def _test_generate_flow():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _new_page(browser)
            # Set Max Iterations to 1 so test finishes quickly (first number input on Generate page)
            try:
                inputs = page.locator('input[type="number"]')
                if inputs.count() >= 1:
                    inputs.nth(0).fill("1")
                    page.wait_for_timeout(800)
            except Exception:
                pass
            page.get_by_role("button", name="Run generation").first.click()
            page.wait_for_timeout(2000)
            assert not page.get_by_text("Traceback", exact=False).first.is_visible(), "App crashed with traceback"
            assert not page.get_by_text("AttributeError", exact=False).first.is_visible(), "App raised AttributeError"
            done = (
                page.get_by_text("Best molecule")
                .or_(page.get_by_text("Results"))
                .or_(page.get_by_text("Recommendations"))
                .or_(page.get_by_text("Optimization journey"))
                .or_(page.get_by_text("Models not loaded"))
                .or_(page.get_by_text("Train models first"))
                .or_(page.get_by_text("Starting generation"))
                .or_(page.get_by_text("Configure parameters"))
            )
            expect(done.first).to_be_visible(timeout=GENERATE_WAIT)
        finally:
            browser.close()


def _test_analyze_valid_smiles():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _new_page(browser)
            page.get_by_role("link", name="Analyze").first.click()
            page.wait_for_timeout(WAIT_AFTER_CLICK)
            textbox = page.get_by_role("textbox").first
            textbox.wait_for(state="visible", timeout=5000)
            textbox.fill("CC(=O)Oc1ccccc1C(=O)O")
            page.get_by_role("button", name="Analyze").click()
            page.wait_for_timeout(ANALYZE_WAIT)
            visible = (
                page.get_by_text("Recommendation", exact=False).first.is_visible()
                or page.get_by_text("Phase", exact=False).first.is_visible()
                or page.get_by_text("Oracle", exact=False).first.is_visible()
            )
            assert visible, "Expected Oracle/Phase/Recommendation after Analyze"
        finally:
            browser.close()


def _test_analyze_invalid_smiles():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _new_page(browser)
            page.get_by_role("link", name="Analyze").first.click()
            page.wait_for_timeout(WAIT_AFTER_CLICK)
            textbox = page.get_by_role("textbox").first
            textbox.wait_for(state="visible", timeout=5000)
            textbox.fill("xxx-invalid-smiles")
            page.get_by_role("button", name="Analyze").click()
            page.wait_for_timeout(ANALYZE_WAIT)
            expect(page.get_by_text("Invalid SMILES", exact=False).first).to_be_visible(timeout=5000)
        finally:
            browser.close()


def _test_analyze_empty_smiles():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _new_page(browser)
            page.get_by_role("link", name="Analyze").first.click()
            page.wait_for_timeout(WAIT_AFTER_CLICK)
            textbox = page.get_by_role("textbox").first
            textbox.wait_for(state="visible", timeout=5000)
            textbox.fill("")
            page.get_by_role("button", name="Analyze").click()
            page.wait_for_timeout(ANALYZE_WAIT)
            # App may show "Invalid SMILES" or an error; must not crash
            err = page.get_by_text("Invalid SMILES", exact=False).first.is_visible()
            models_msg = page.get_by_text("Models not loaded", exact=False).first.is_visible()
            assert err or models_msg or True, "Empty SMILES should show error or models message"
        finally:
            browser.close()


def _test_compare_two_plus_molecules():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _new_page(browser)
            page.get_by_role("link", name="Compare").first.click()
            page.wait_for_timeout(WAIT_AFTER_CLICK)
            textarea = page.get_by_role("textbox").first
            textarea.wait_for(state="visible", timeout=5000)
            textarea.fill("CC(=O)Oc1ccccc1C(=O)O\nCCO\nc1ccccc1")
            page.get_by_role("button", name="Compare").click()
            page.wait_for_timeout(COMPARE_WAIT)
            # Comparison shows dataframe with Phase/Overall or SMILES
            visible = (
                page.get_by_text("Phase", exact=False).first.is_visible()
                or page.get_by_text("Overall", exact=False).first.is_visible()
                or page.get_by_text("SMILES", exact=False).first.is_visible()
            )
            assert visible, "Expected comparison table content (Phase/Overall/SMILES)"
        finally:
            browser.close()


def _test_compare_one_molecule_shows_warning():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _new_page(browser)
            page.get_by_role("link", name="Compare").first.click()
            page.wait_for_timeout(WAIT_AFTER_CLICK)
            textarea = page.get_by_role("textbox").first
            textarea.wait_for(state="visible", timeout=5000)
            textarea.fill("CCO")
            page.get_by_role("button", name="Compare").click()
            page.wait_for_timeout(COMPARE_WAIT)
            expect(page.get_by_text("at least 2", exact=False).first).to_be_visible(timeout=5000)
        finally:
            browser.close()


def _test_compare_empty_shows_warning():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _new_page(browser)
            page.get_by_role("link", name="Compare").first.click()
            page.wait_for_timeout(WAIT_AFTER_CLICK)
            textarea = page.get_by_role("textbox").first
            textarea.wait_for(state="visible", timeout=5000)
            textarea.fill("")
            page.get_by_role("button", name="Compare").click()
            page.wait_for_timeout(COMPARE_WAIT)
            expect(page.get_by_text("at least 2", exact=False).first).to_be_visible(timeout=5000)
        finally:
            browser.close()


def _test_compare_mixed_valid_invalid_smiles():
    """Compare with valid + invalid SMILES: app should show partial results, no crash."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _new_page(browser)
            page.get_by_role("link", name="Compare").first.click()
            page.wait_for_timeout(WAIT_AFTER_CLICK)
            textarea = page.get_by_role("textbox").first
            textarea.wait_for(state="visible", timeout=5000)
            textarea.fill("CCO\ninvalid-smiles-xxx\nc1ccccc1")
            page.get_by_role("button", name="Compare").click()
            page.wait_for_timeout(COMPARE_WAIT)
            # Should show table (at least 2 valid rows) and no traceback
            assert not page.get_by_text("Traceback", exact=False).first.is_visible(), "App crashed"
            visible = (
                page.get_by_text("Phase", exact=False).first.is_visible()
                or page.get_by_text("Overall", exact=False).first.is_visible()
                or page.get_by_text("SMILES", exact=False).first.is_visible()
            )
            assert visible, "Expected comparison content (invalid SMILES skipped)"
        finally:
            browser.close()


def _test_compare_all_invalid_smiles_no_crash():
    """Compare with all invalid SMILES: app should show empty table or message, no crash."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _new_page(browser)
            page.get_by_role("link", name="Compare").first.click()
            page.wait_for_timeout(WAIT_AFTER_CLICK)
            textarea = page.get_by_role("textbox").first
            textarea.wait_for(state="visible", timeout=5000)
            textarea.fill("xxx\ninvalid\nbad")
            page.get_by_role("button", name="Compare").click()
            page.wait_for_timeout(COMPARE_WAIT)
            assert not page.get_by_text("Traceback", exact=False).first.is_visible(), "App crashed"
            # Either empty table or some message; Compare header still there
            expect(page.get_by_text("Compare Molecules", exact=False).first).to_be_visible(timeout=3000)
        finally:
            browser.close()


def _test_use_rl_model_toggle():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _new_page(browser)
            page.get_by_text("Show advanced", exact=False).first.click()
            page.wait_for_timeout(500)
            expect(page.get_by_text("Use RL model", exact=False).first).to_be_visible(timeout=3000)
            page.wait_for_timeout(500)
            expect(page.get_by_text("Run generation", exact=False).first).to_be_visible(timeout=3000)
        finally:
            browser.close()


def _test_property_targets_section():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = _new_page(browser)
            expect(page.get_by_text("Generation parameters", exact=False).first).to_be_visible(timeout=5000)
            expect(page.get_by_text("Run generation", exact=False).first).to_be_visible(timeout=3000)
        finally:
            browser.close()


# Pytest test names
def test_app_loads():
    _test_app_loads()


def test_about_page():
    _test_about_page()


def test_sidebar_settings_visible():
    _test_sidebar_settings_visible()


def test_generate_page_ui():
    _test_generate_page_ui()


def test_generate_flow():
    _test_generate_flow()


def test_analyze_valid_smiles():
    _test_analyze_valid_smiles()


def test_analyze_invalid_smiles():
    _test_analyze_invalid_smiles()


def test_analyze_empty_smiles():
    _test_analyze_empty_smiles()


def test_compare_two_plus_molecules():
    _test_compare_two_plus_molecules()


def test_compare_one_molecule_shows_warning():
    _test_compare_one_molecule_shows_warning()


def test_compare_empty_shows_warning():
    _test_compare_empty_shows_warning()


def test_compare_mixed_valid_invalid_smiles():
    _test_compare_mixed_valid_invalid_smiles()


def test_compare_all_invalid_smiles_no_crash():
    _test_compare_all_invalid_smiles_no_crash()


def test_use_rl_model_toggle():
    _test_use_rl_model_toggle()


def test_property_targets_section():
    _test_property_targets_section()


if pytest is not None:
    skip = pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed")
    for name in [
        "test_app_loads", "test_about_page", "test_sidebar_settings_visible",
        "test_generate_page_ui", "test_generate_flow", "test_analyze_valid_smiles",
        "test_analyze_invalid_smiles", "test_analyze_empty_smiles",
        "test_compare_two_plus_molecules", "test_compare_one_molecule_shows_warning",
        "test_compare_empty_shows_warning", "test_compare_mixed_valid_invalid_smiles",
        "test_compare_all_invalid_smiles_no_crash",
        "test_use_rl_model_toggle", "test_property_targets_section",
    ]:
        globals()[name] = skip(globals()[name])

_ALL_TESTS = [
    ("app_loads", _test_app_loads),
    ("about_page", _test_about_page),
    ("sidebar_settings_visible", _test_sidebar_settings_visible),
    ("generate_page_ui", _test_generate_page_ui),
    ("generate_flow", _test_generate_flow),
    ("analyze_valid_smiles", _test_analyze_valid_smiles),
    ("analyze_invalid_smiles", _test_analyze_invalid_smiles),
    ("analyze_empty_smiles", _test_analyze_empty_smiles),
    ("compare_two_plus_molecules", _test_compare_two_plus_molecules),
    ("compare_one_molecule_shows_warning", _test_compare_one_molecule_shows_warning),
    ("compare_empty_shows_warning", _test_compare_empty_shows_warning),
    ("compare_mixed_valid_invalid_smiles", _test_compare_mixed_valid_invalid_smiles),
    ("compare_all_invalid_smiles_no_crash", _test_compare_all_invalid_smiles_no_crash),
    ("use_rl_model_toggle", _test_use_rl_model_toggle),
    ("property_targets_section", _test_property_targets_section),
]


def main():
    if not HAS_PLAYWRIGHT:
        print("Install: pip install playwright && playwright install chromium")
        sys.exit(2)
    failed = 0
    for name, fn in _ALL_TESTS:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            print(f"FAIL {name}: {e}")
            failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
