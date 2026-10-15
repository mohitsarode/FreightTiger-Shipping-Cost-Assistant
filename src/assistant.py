import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config import DEFAULT_CONFIG
from src.vector_store import ContextNotesStore

class FreightCostAssistant:
    """
    Stretch Goal: Conversational Assistant that answers user questions
    about route cost trends, anomalies, and justifications in plain English.
    """
    
    def __init__(self, output_csv_path: Optional[Path] = None, all_weeks_csv_path: Optional[Path] = None):
        self.output_csv_path = output_csv_path or DEFAULT_CONFIG.output_path
        self.all_weeks_csv_path = all_weeks_csv_path or DEFAULT_CONFIG.output_all_weeks_path
        
        self.anomalies_df = pd.read_csv(self.output_csv_path, dtype=str).fillna("")
        if self.all_weeks_csv_path.exists():
            self.all_weeks_df = pd.read_csv(self.all_weeks_csv_path, dtype=str).fillna("")
        else:
            self.all_weeks_df = self.anomalies_df
            
        self.notes_store = ContextNotesStore(DEFAULT_CONFIG.context_notes_path)
        
    def answer_question(self, query: str) -> str:
        """
        Interprets natural language question and answers using the system's own findings.
        """
        q = query.lower()
        
        # 1. Query: List justified routes/anomalies
        if "justified" in q and ("list" in q or "which" in q or "all" in q or "show" in q):
            justified = self.anomalies_df[self.anomalies_df['flagged'] == 'No (justified)']
            if justified.empty:
                return "There are no justified cost anomalies recorded."
            lines = ["Here are the route anomalies with confirmed valid justifications:"]
            for _, r in justified.iterrows():
                lines.append(
                    f"• {r['route']} (Week of {r['week_of']}): Cost {r['cost_per_tonne_km']} INR/t-km "
                    f"({r['vs_own_history']}, {r['vs_similar_routes']}). "
                    f"Justified by Note {r['matched_note_id']}: {r['reason']}"
                )
            return "\n\n".join(lines)

        # 2. Query: List unexplained / flagged routes
        if ("unexplained" in q or "flagged" in q or "human review" in q) and ("list" in q or "which" in q or "all" in q or "show" in q):
            unexplained = self.anomalies_df[self.anomalies_df['flagged'] == 'Yes']
            lines = [f"Found {len(unexplained)} unexplained cost anomalies flagged for review:"]
            # Group by route
            for route, grp in unexplained.groupby('route'):
                lines.append(f"**Route {route}** ({len(grp)} weeks flagged):")
                for _, r in grp.iterrows():
                    lines.append(f"  - Week {r['week_of']}: {r['cost_per_tonne_km']} INR/t-km ({r['vs_own_history']}, {r['vs_similar_routes']})")
            return "\n".join(lines)

        # 3. Route-specific inquiry
        # Detect route mentioned in query
        matched_route = None
        for r in self.all_weeks_df['route'].unique():
            parts = r.lower().split('-')
            if r.lower() in q or (parts[0] in q and parts[1] in q):
                matched_route = r
                break
                
        # Detect month/date mentioned in query
        months = {
            "january": "01", "jan": "01",
            "february": "02", "feb": "02",
            "march": "03", "mar": "03",
            "april": "04", "apr": "04",
            "may": "05",
            "june": "06", "jun": "06",
            "july": "07", "jul": "07",
            "august": "08", "aug": "08",
            "september": "09", "sep": "09", "sept": "09",
            "october": "10", "oct": "10",
            "november": "11", "nov": "11",
            "december": "12", "dec": "12"
        }
        
        matched_month = None
        for m_name, m_num in months.items():
            if re.search(r'\b' + m_name + r'\b', q):
                matched_month = m_num
                break
                
        matched_year = None
        year_match = re.search(r'\b(2024|2025)\b', q)
        if year_match:
            matched_year = year_match.group(1)
            
        # If route was detected:
        if matched_route:
            sub = self.all_weeks_df[self.all_weeks_df['route'] == matched_route]
            if matched_year:
                sub = sub[sub['week_of'].str.startswith(matched_year)]
            if matched_month:
                sub = sub[sub['week_of'].str.slice(5, 7) == matched_month]
                
            if sub.empty:
                return f"No records found for route '{matched_route}' matching specified time period ({matched_month or ''}/{matched_year or ''})."
                
            # Check if any anomalies in this filtered set
            sub_anomalies = sub[sub['flagged'].isin(['Yes', 'No (justified)'])]
            if not sub_anomalies.empty:
                lines = [f"Analysis for **{matched_route}** during this period:"]
                for _, row in sub_anomalies.iterrows():
                    verdict_str = "JUSTIFIED" if row['flagged'] == 'No (justified)' else "UNEXPLAINED (Flagged)"
                    lines.append(
                        f"• **Week of {row['week_of']}** [{verdict_str}]:\n"
                        f"  Cost: {row['cost_per_tonne_km']} INR/t-km ({row['vs_own_history']}, {row['vs_similar_routes']}).\n"
                        f"  Explanation: {row['reason']}"
                    )
                return "\n\n".join(lines)
            else:
                avg_cost = sub['cost_per_tonne_km'].astype(float).mean()
                return (
                    f"During this period on route **{matched_route}**, costs remained normal with an average of "
                    f"{avg_cost:.2f} INR/tonne-km. No anomalies or unjustified cost surges were detected."
                )

        # 4. Highest cost or summary statistics
        if "highest" in q or "maximum" in q or "most expensive" in q:
            df_numeric = self.all_weeks_df.copy()
            df_numeric['cost_num'] = df_numeric['cost_per_tonne_km'].astype(float)
            top_row = df_numeric.sort_values('cost_num', ascending=False).iloc[0]
            return (
                f"The highest cost per tonne-km in the entire dataset was **{top_row['cost_num']:.2f} INR/tonne-km** "
                f"on route **{top_row['route']}** during the week of **{top_row['week_of']}** "
                f"({top_row['vs_own_history']}, {top_row['vs_similar_routes']}). "
                f"Verdict: {top_row['flagged']}."
            )

        # Fallback summary response
        total_anomalies = len(self.anomalies_df)
        justified_count = (self.anomalies_df['flagged'] == 'No (justified)').sum()
        unexplained_count = (self.anomalies_df['flagged'] == 'Yes').sum()
        
        return (
            f"FreightTiger Assistant Summary:\n"
            f"• Monitored 7 corridors across Short, Medium, and Long route types over 104 weeks.\n"
            f"• Detected {total_anomalies} candidate cost anomalies:\n"
            f"  - {justified_count} justified by verified operational events (e.g. floods, festive surcharges).\n"
            f"  - {unexplained_count} unexplained cost surges flagged for human review (including late-2025 creep on Mumbai-Pune).\n\n"
            f"You can ask me questions like:\n"
            f"- 'Why did Ahmedabad-Mumbai get pricier in January 2025?'\n"
            f"- 'Why did Chennai-Bangalore surge in March 2025?'\n"
            f"- 'Why was Mumbai-Pune flagged in September 2025?'\n"
            f"- 'List all justified routes'\n"
            f"- 'What was the highest cost route?'"
        )
