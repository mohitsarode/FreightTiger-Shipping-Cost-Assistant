import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContextNotesStore:
    """
    RAG Vector Store and Retrieval Layer for Context Notes.
    Indexes notes with TF-IDF embeddings and provides hybrid search:
    semantic text similarity + corridor filtering + temporal proximity.
    """
    def __init__(self, notes_csv_path: Path):
        self.notes_csv_path = notes_csv_path
        self.df = pd.read_csv(notes_csv_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['route'] = self.df['applies_to'].str.strip()
        self.df['note_text'] = self.df['note'].str.strip()
        
        # Build search corpus combining route metadata and note content
        self.corpus = [
            f"Note {r['note_id']} on {r['date'].strftime('%Y-%m-%d')} for {r['applies_to']}: {r['note']}"
            for _, r in self.df.iterrows()
        ]
        
        # Vectorize corpus
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        
    def retrieve(
        self,
        route: str,
        week_of: str,
        query_text: Optional[str] = None,
        max_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieves candidate context notes relevant to a given route and week.
        Scores notes based on semantic similarity, corridor match, and temporal proximity.
        """
        target_date = pd.to_datetime(week_of)
        results = []
        
        # Query vector
        full_query = f"Route {route} week {week_of}. Cost increase surcharge fuel disruption weather. {query_text or ''}"
        query_vec = self.vectorizer.transform([full_query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        for idx, row in self.df.iterrows():
            note_date = row['date']
            days_diff = abs((target_date - note_date).days)
            corridor = row['applies_to']
            
            # Corridor match score
            is_direct_corridor = (corridor.lower() == route.lower())
            is_all_routes = (corridor.lower() == 'all routes')
            corridor_score = 1.0 if is_direct_corridor else (0.5 if is_all_routes else 0.0)
            
            # Temporal proximity score (exponential decay over weeks)
            # If within 7 days: ~1.0; if 30 days: ~0.4; if 180 days: ~0.02
            time_score = float(np.exp(-days_diff / 30.0))
            
            # Semantic score from TF-IDF
            sem_score = float(sim_scores[idx])
            
            # Combined hybrid retrieval score
            # Strongly prioritize corridor match and temporal proximity
            combined_score = (corridor_score * 0.45) + (time_score * 0.35) + (sem_score * 0.20)
            
            results.append({
                "note_id": row['note_id'],
                "date": note_date.strftime('%Y-%m-%d'),
                "applies_to": corridor,
                "note": row['note_text'],
                "days_diff": days_diff,
                "is_direct_corridor": is_direct_corridor,
                "is_all_routes": is_all_routes,
                "corridor_score": corridor_score,
                "time_score": time_score,
                "semantic_score": sem_score,
                "combined_score": combined_score
            })
            
        # Sort by combined score descending
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:max_results]
    
    def get_note_by_id(self, note_id: str) -> Optional[Dict[str, Any]]:
        sub = self.df[self.df['note_id'] == note_id]
        if sub.empty:
            return None
        r = sub.iloc[0]
        return {
            "note_id": r['note_id'],
            "date": r['date'].strftime('%Y-%m-%d'),
            "applies_to": r['applies_to'],
            "note": r['note_text']
        }
