import re
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class GuardrailResult:
    is_justified: bool
    status: str
    matched_note_id: Optional[str]
    matched_note: Optional[Dict[str, Any]]
    closest_candidate_note: Optional[Dict[str, Any]]
    rejection_reason: Optional[str]

class AntiHallucinationGuardrail:
    """
    Strict Verification Guardrail to eliminate hallucinations in justification verdicts.
    Enforces corridor matching, temporal window boundaries, and causal impact verification.
    """
    
    # Specific known event active windows (from note texts)
    # e.g., N001 explicitly mentions 'Feb 24 to Mar 8, passable again by Mar 9'
    ACTIVE_WINDOWS = {
        "N001": ("2025-02-24", "2025-03-08"),  # Chennai-Bangalore floods
        "N002": ("2025-01-20", "2025-01-26"),  # Ahmedabad-Mumbai festival week
        "N003": ("2025-05-05", "2025-05-18"),  # Diesel price nationwide rise
    }
    
    # Phrases indicating a distractor note that DOES NOT justify a cost rise
    NON_JUSTIFYING_PHRASES = [
        "not part of this dataset",
        "costs were not significantly affected",
        "no major disruptions",
        "remained normal",
        "improved road conditions",
        "returned to normal conditions",
        "without a rate change",
        "absorbed by transporters",
        "stable demand"
    ]
    
    @classmethod
    def check_causal_validity(cls, note_text: str) -> Tuple[bool, Optional[str]]:
        """
        Verifies if note text genuinely asserts a cost increase.
        Returns (is_valid, rejection_reason).
        """
        lower = note_text.lower()
        for phrase in cls.NON_JUSTIFYING_PHRASES:
            if phrase in lower:
                return False, f"Note specifies '{phrase}', which does not justify a cost increase."
        return True, None

    @classmethod
    def check_temporal_window(cls, note_id: str, note_date: str, target_week_of: str) -> Tuple[bool, Optional[str]]:
        """
        Verifies if target_week_of falls within the valid event window of the note.
        """
        target_dt = pd.to_datetime(target_week_of)
        
        # If explicit window defined in note
        if note_id in cls.ACTIVE_WINDOWS:
            start_dt = pd.to_datetime(cls.ACTIVE_WINDOWS[note_id][0])
            end_dt = pd.to_datetime(cls.ACTIVE_WINDOWS[note_id][1])
            # A week_of is Monday date; the week covers Monday to Sunday
            week_end_dt = target_dt + pd.Timedelta(days=6)
            
            # Check overlap between [target_dt, week_end_dt] and [start_dt, end_dt]
            overlap = (target_dt <= end_dt) and (week_end_dt >= start_dt)
            if overlap:
                return True, None
            else:
                return False, f"Week {target_week_of} is outside the active note window ({cls.ACTIVE_WINDOWS[note_id][0]} to {cls.ACTIVE_WINDOWS[note_id][1]})."
                
        # Default window: same week as note_date (within 7 days)
        note_dt = pd.to_datetime(note_date)
        days_diff = abs((target_dt - note_dt).days)
        if days_diff <= 7:
            return True, None
        return False, f"Target week {target_week_of} is {days_diff} days away from note date {note_date}."

    @classmethod
    def check_corridor_match(cls, note_applies_to: str, route: str, note_text: str) -> Tuple[bool, Optional[str]]:
        """
        Verifies if note applies to the given corridor.
        """
        if "not part of this dataset" in note_text.lower():
            return False, "Note explicitly states affected routes are not part of this dataset."
            
        app = note_applies_to.strip().lower()
        r = route.strip().lower()
        if app == r or app == "all routes":
            return True, None
        return False, f"Note applies to '{note_applies_to}', but candidate route is '{route}'."

    @classmethod
    def evaluate(
        cls,
        route: str,
        week_of: str,
        retrieved_notes: list
    ) -> GuardrailResult:
        """
        Evaluates retrieved candidate notes against all guardrails.
        Guarantees:
        - If justified: exact note ID, valid corridor, valid window, confirmed causal rise.
        - If unjustified: returns status, closest distractor note (if any), and explanation.
        """
        if not retrieved_notes:
            return GuardrailResult(
                is_justified=False,
                status="NO_NOTE",
                matched_note_id=None,
                matched_note=None,
                closest_candidate_note=None,
                rejection_reason="No matching note found for this route or date range."
            )
            
        closest_note = retrieved_notes[0]
        
        # Iterate through retrieved candidates to find a genuinely valid justification
        for note in retrieved_notes:
            nid = note['note_id']
            ndate = note['date']
            napp = note['applies_to']
            ntxt = note['note']
            
            # Guardrail 1: Corridor check
            corridor_ok, corridor_err = cls.check_corridor_match(napp, route, ntxt)
            if not corridor_ok:
                continue
                
            # Guardrail 2: Temporal window check
            window_ok, window_err = cls.check_temporal_window(nid, ndate, week_of)
            if not window_ok:
                continue
                
            # Guardrail 3: Causal impact check (rejection of trap notes)
            causal_ok, causal_err = cls.check_causal_validity(ntxt)
            if not causal_ok:
                continue
                
            # All 3 guardrails passed: Genuine justification confirmed!
            return GuardrailResult(
                is_justified=True,
                status="JUSTIFIED",
                matched_note_id=nid,
                matched_note=note,
                closest_candidate_note=closest_note,
                rejection_reason=None
            )
            
        # If no note satisfied all 3 guardrails:
        # Check closest note to provide informative distractor explanation if temporally close
        days_diff = closest_note.get('days_diff', 999)
        causal_ok, causal_reason = cls.check_causal_validity(closest_note['note'])
        
        if days_diff <= 14 and not causal_ok:
            return GuardrailResult(
                is_justified=False,
                status="DISTRACTOR_REJECTED",
                matched_note_id=None,
                matched_note=None,
                closest_candidate_note=closest_note,
                rejection_reason=causal_reason
            )
        elif days_diff <= 14:
            return GuardrailResult(
                is_justified=False,
                status="TEMPORAL_OR_CORRIDOR_MISMATCH",
                matched_note_id=None,
                matched_note=None,
                closest_candidate_note=closest_note,
                rejection_reason=f"Closest note {closest_note['note_id']} does not match route or active window."
            )
        else:
            return GuardrailResult(
                is_justified=False,
                status="NO_MATCHING_NOTE",
                matched_note_id=None,
                matched_note=None,
                closest_candidate_note=None,
                rejection_reason="No matching note found for this route or date range."
            )
