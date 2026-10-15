import pytest
from src.guardrails import AntiHallucinationGuardrail

def test_trap_notes_are_rejected():
    trap_cases = [
        ("N004", "A new toll plaza was commissioned, but affected routes are not part of this dataset."),
        ("N005", "Highway maintenance caused minor delays for about a week; costs were not significantly affected."),
        ("N006", "Overall demand remained stable with no major disruptions reported."),
        ("N007", "Improved road conditions after resurfacing work was completed."),
        ("N008", "No significant disruptions reported; freight movement remained normal."),
        ("N009", "Highway returned to normal conditions after flood repairs completed."),
        ("N010", "Compliance costs were absorbed by transporters without a rate change.")
    ]
    for nid, text in trap_cases:
        is_valid, reason = AntiHallucinationGuardrail.check_causal_validity(text)
        assert not is_valid, f"Trap note {nid} should have been rejected! Reason: {reason}"

def test_valid_causal_notes_are_accepted():
    valid_cases = [
        ("N001", "Heavy flooding disrupted normal truck movement forcing longer detours and higher trip costs."),
        ("N002", "A regional festival week saw a temporary surcharge applied by transporters due to high demand."),
        ("N003", "Diesel prices rose nationwide starting this week, pushing up transportation costs across all routes.")
    ]
    for nid, text in valid_cases:
        is_valid, reason = AntiHallucinationGuardrail.check_causal_validity(text)
        assert is_valid, f"Valid note {nid} was incorrectly rejected! Reason: {reason}"

def test_temporal_window_boundaries():
    # N001 active window is Feb 24 to Mar 8, 2025
    # Week of 2025-02-24 (Feb 24 - Mar 2) -> Should PASS
    ok, err = AntiHallucinationGuardrail.check_temporal_window("N001", "2025-02-24", "2025-02-24")
    assert ok, f"Expected week 2025-02-24 to pass for N001: {err}"
    
    # Week of 2025-03-03 (Mar 3 - Mar 9) -> Should PASS
    ok, err = AntiHallucinationGuardrail.check_temporal_window("N001", "2025-02-24", "2025-03-03")
    assert ok, f"Expected week 2025-03-03 to pass for N001: {err}"
    
    # Week of 2025-04-14 -> Should FAIL
    ok, err = AntiHallucinationGuardrail.check_temporal_window("N001", "2025-02-24", "2025-04-14")
    assert not ok, "Expected week 2025-04-14 to be rejected for N001"

def test_corridor_matching():
    # Exact corridor match
    ok, err = AntiHallucinationGuardrail.check_corridor_match("Chennai-Bangalore", "Chennai-Bangalore", "Heavy flooding")
    assert ok
    
    # All Routes matches any corridor
    ok, err = AntiHallucinationGuardrail.check_corridor_match("All Routes", "Delhi-Jaipur", "Fuel price increase")
    assert ok
    
    # Wrong corridor rejected
    ok, err = AntiHallucinationGuardrail.check_corridor_match("Chennai-Bangalore", "Mumbai-Delhi", "Heavy flooding")
    assert not ok
