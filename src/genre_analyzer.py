"""
src/genre_analyzer.py

Genre-Aware Contextual Toxicity Analysis.

Provides:
  - Genre identification and normalization
  - Non-binding genre suggestion from video metadata
  - Genre-conditioned probability adjustment (supplementary context only)
  - Strong-evidence guard: never overrides threat/severe_toxic when prob >= 0.78

DESIGN PRINCIPLE:
  Genre is supplementary CONTEXT, never the primary decision-maker.
  Strong abuse/threat/obscenity evidence ALWAYS overrides genre mitigations.
  Emojis alone are NEVER sufficient evidence of toxicity.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple

# ---------------------------------------------------------------------------
# Genre registry
# ---------------------------------------------------------------------------

SUPPORTED_GENRES: List[str] = [
    "Comedy",
    "Entertainment",
    "Education",
    "Politics",
    "Gaming",
    "Music",
    "Sports",
    "News",
    "Technology",
    "Movies",
    "Sensitive / Social Issues",
    "General",
]

# Alias map — lower-case input to canonical genre name
GENRE_ALIASES: Dict[str, str] = {
    "comedy": "Comedy",
    "humor": "Comedy",
    "humour": "Comedy",
    "funny": "Comedy",
    "entertainment": "Entertainment",
    "entertain": "Entertainment",
    "fun": "Entertainment",
    "lifestyle": "Entertainment",
    "education": "Education",
    "educational": "Education",
    "learning": "Education",
    "tutorial": "Education",
    "howto": "Education",
    "how-to": "Education",
    "science": "Education",
    "lecture": "Education",
    "politics": "Politics",
    "political": "Politics",
    "government": "Politics",
    "election": "Politics",
    "policy": "Politics",
    "gaming": "Gaming",
    "game": "Gaming",
    "games": "Gaming",
    "esports": "Gaming",
    "livestream": "Gaming",
    "lets play": "Gaming",
    "let's play": "Gaming",
    "music": "Music",
    "song": "Music",
    "songs": "Music",
    "audio": "Music",
    "lyrics": "Music",
    "album": "Music",
    "rap": "Music",
    "pop": "Music",
    "sports": "Sports",
    "sport": "Sports",
    "football": "Sports",
    "cricket": "Sports",
    "basketball": "Sports",
    "soccer": "Sports",
    "athletics": "Sports",
    "news": "News",
    "breaking": "News",
    "current events": "News",
    "journalism": "News",
    "technology": "Technology",
    "tech": "Technology",
    "programming": "Technology",
    "coding": "Technology",
    "software": "Technology",
    "ai": "Technology",
    "movies": "Movies",
    "movie": "Movies",
    "film": "Movies",
    "cinema": "Movies",
    "review": "Movies",
    "trailer": "Movies",
    "sensitive": "Sensitive / Social Issues",
    "social issues": "Sensitive / Social Issues",
    "social": "Sensitive / Social Issues",
    "awareness": "Sensitive / Social Issues",
    "general": "General",
    "auto": "General",
    "": "General",
}

# ---------------------------------------------------------------------------
# Metadata hint keywords for auto-suggestion (non-binding)
# ---------------------------------------------------------------------------

GENRE_METADATA_HINTS: Dict[str, List[str]] = {
    "Comedy": [
        "comedy", "funny", "humor", "humour", "laugh", "prank", "roast",
        "standup", "stand-up", "meme", "jokes", "hilarious", "sketch",
        "parody", "satirical", "fun with"
    ],
    "Education": [
        "tutorial", "how to", "learn", "lecture", "course", "lesson",
        "explained", "explanation", "guide", "tips", "tricks", "study",
        "education", "science", "math", "history", "chemistry", "physics",
        "biology", "language", "grammar"
    ],
    "Politics": [
        "politics", "political", "election", "vote", "government",
        "parliament", "policy", "minister", "president", "congress",
        "senate", "protest", "democracy", "debate"
    ],
    "Gaming": [
        "gaming", "gameplay", "game", "playthrough", "walkthrough",
        "esports", "streamer", "twitch", "minecraft", "pubg", "fortnite",
        "gta", "cod", "valorant", "speedrun", "pro player"
    ],
    "Music": [
        "music", "song", "lyric", "album", "official video", "mv",
        "audio", "single", "ft.", "feat.", "rap", "beat", "playlist",
        "concert", "live performance", "music video", "ost"
    ],
    "Sports": [
        "cricket", "football", "soccer", "basketball", "tennis", "hockey",
        "ipl", "nba", "fifa", "match", "tournament", "championship",
        "league", "highlight", "goal", "wicket", "stadium", "athlete"
    ],
    "News": [
        "news", "breaking", "update", "report", "today", "current",
        "latest", "journalism", "press", "media", "coverage", "anchor",
        "journalist", "bbc", "cnn", "ndtv", "republic"
    ],
    "Technology": [
        "technology", "tech", "programming", "coding", "software",
        "developer", "python", "javascript", "ai", "machine learning",
        "deep learning", "chatgpt", "android", "iphone", "review",
        "unboxing", "laptop", "smartphone", "gadget"
    ],
    "Movies": [
        "movie", "film", "cinema", "trailer", "review", "bollywood",
        "hollywood", "kollywood", "sandalwood", "netflix", "ott",
        "web series", "series", "episode", "actor", "actress", "director",
        "scene", "climax", "sequel"
    ],
    "Entertainment": [
        "entertainment", "vlog", "lifestyle", "daily", "reaction",
        "unboxing", "challenge", "trend", "shorts", "reels", "tiktok",
        "instagram", "fun", "awesome", "viral", "pov"
    ],
    "Sensitive / Social Issues": [
        "mental health", "depression", "suicide", "awareness", "social issues",
        "discrimination", "racism", "violence", "harassment", "lgbtq",
        "gender", "feminism", "abuse", "trauma", "trigger warning",
        "sensitive", "suicide prevention"
    ],
}


def normalize_genre(genre: Optional[str]) -> str:
    """
    Normalize raw genre input to a canonical SUPPORTED_GENRES name.
    Falls back to 'General' for unrecognized inputs.
    """
    if not genre:
        return "General"
    cleaned = genre.strip().lower()
    if cleaned in GENRE_ALIASES:
        return GENRE_ALIASES[cleaned]
    for g in SUPPORTED_GENRES:
        if g.lower() == cleaned:
            return g
    return "General"


def genre_to_feature_token(genre: str) -> str:
    """
    Convert a canonical genre name to a TF-IDF feature token.
    E.g. 'Comedy' returns '__genre_comedy__'
    """
    canonical = normalize_genre(genre)
    slug = canonical.lower().replace(" / ", "_").replace(" ", "_").replace("/", "_")
    return f"__genre_{slug}__"


def suggest_video_genre(
    title: str = "",
    description: str = "",
    tags: Optional[List[str]] = None,
) -> Tuple[str, float]:
    """
    Non-binding genre suggestion from video title, description, and tags.
    Returns (suggested_genre, confidence_score 0.0-1.0).
    Returns ('General', 0.0) if no genre can be determined confidently.
    """
    combined = " ".join([
        (title or "").lower(),
        (description or "")[:300].lower(),
        " ".join((tags or []))[:200].lower()
    ])

    scores: Dict[str, int] = {g: 0 for g in SUPPORTED_GENRES if g != "General"}
    for genre, keywords in GENRE_METADATA_HINTS.items():
        for kw in keywords:
            if kw in combined:
                scores[genre] = scores.get(genre, 0) + 1

    if not scores:
        return "General", 0.0

    best_genre = max(scores, key=lambda g: scores[g])
    best_score = scores[best_genre]

    if best_score == 0:
        return "General", 0.0

    confidence = min(1.0, round(best_score / 5.0, 2))
    return best_genre, confidence


# ---------------------------------------------------------------------------
# Genre Context Evaluator
# ---------------------------------------------------------------------------

# Strong-evidence threshold: genre mitigations are SKIPPED when any
# probability reaches this level.
_STRONG_EVIDENCE_THRESHOLD = 0.78


class GenreContextEvaluator:
    """
    Applies genre-conditioned probability adjustments as supplementary context.

    Rules:
      - Strong-evidence guard fires first: if any prob >= 0.78 OR the comment
        contains a directed_attack or violent_threat signal, genre context is
        completely skipped.
      - Genre mitigations reduce borderline probabilities.
      - Genre amplifications increase probabilities for clearly hostile content
        in sensitive contexts.
      - Every adjustment is capped so it cannot flip a confident prediction.
    """

    # Genre-specific mitigation caps: probs above these values will NOT be
    # mitigated by genre context.
    _MITIGATION_CAPS: Dict[str, float] = {
        "Comedy":                    0.60,
        "Entertainment":             0.60,
        "Education":                 0.55,
        "Gaming":                    0.60,
        "Music":                     0.60,
        "Sports":                    0.65,
        "Movies":                    0.65,
        "Politics":                  0.72,
        "News":                      0.72,
        "Sensitive / Social Issues": 0.50,
        "General":                   1.00,
    }

    # Mitigation strength (factor by which borderline probs are reduced)
    _MITIGATION_FACTOR: Dict[str, float] = {
        "Comedy":                    0.40,
        "Entertainment":             0.40,
        "Education":                 0.35,
        "Gaming":                    0.40,
        "Music":                     0.40,
        "Sports":                    0.30,
        "Movies":                    0.30,
        "Politics":                  0.20,
        "News":                      0.20,
        "Sensitive / Social Issues": 0.00,
        "General":                   0.00,
    }

    _SENSITIVE_AMPLIFY_LABELS: List[str] = [
        "identity_hate", "insult", "severe_toxic"
    ]

    def evaluate(
        self,
        genre: str,
        adjusted_probs: Dict[str, float],
        flagged_labels: List[str],
        is_directed_attack: bool = False,
        is_violent_threat: bool = False,
        language_code: str = "english",
    ) -> Dict[str, Any]:
        """
        Apply genre-conditioned probability adjustments.

        Returns a dict with:
          genre_probs      : adjusted probabilities after genre context
          genre_flags      : updated flagged labels
          genre_adjusted   : True if any adjustment was made
          genre_reason     : human-readable explanation
          genre_skipped    : True if strong-evidence guard fired
        """
        canonical = normalize_genre(genre)
        probs = dict(adjusted_probs)
        flags = list(flagged_labels)
        reason_parts: List[str] = []

        # 1. Strong-evidence guard
        max_prob = max(probs.values(), default=0.0)
        strong_evidence = (
            max_prob >= _STRONG_EVIDENCE_THRESHOLD
            or is_directed_attack
            or is_violent_threat
        )
        if strong_evidence:
            return {
                "genre_probs": probs,
                "genre_flags": flags,
                "genre_adjusted": False,
                "genre_skipped": True,
                "genre_reason": (
                    "Genre context skipped — strong abuse/threat evidence detected "
                    "(max_prob={:.2f}, directed_attack={}, violent_threat={}).".format(
                        max_prob, is_directed_attack, is_violent_threat
                    )
                )
            }

        if canonical == "General":
            return {
                "genre_probs": probs,
                "genre_flags": flags,
                "genre_adjusted": False,
                "genre_skipped": False,
                "genre_reason": "General genre — no genre-specific adjustment applied."
            }

        cap = self._MITIGATION_CAPS.get(canonical, 1.0)
        factor = self._MITIGATION_FACTOR.get(canonical, 0.0)
        adjusted = False

        # 2. Sensitive / Social Issues — amplify cruelty/hate signals
        if canonical == "Sensitive / Social Issues":
            for label in self._SENSITIVE_AMPLIFY_LABELS:
                if label in probs:
                    old = probs[label]
                    new_p = min(1.0, round(old * 1.15, 4))
                    if new_p != old:
                        probs[label] = new_p
                        adjusted = True
            if adjusted:
                reason_parts.append(
                    "Sensitive/Social Issues context: cruelty and hate signals amplified by 15%"
                )

        # 3. Apply mitigation for non-hostile borderline cases
        elif factor > 0.0:
            mitigation_labels = ["toxic", "insult", "obscene"]
            for label in mitigation_labels:
                if label not in probs:
                    continue
                p = probs[label]
                if p < cap:
                    new_p = round(p * (1.0 - factor), 4)
                    probs[label] = new_p
                    if label in flags and new_p < 0.35:
                        flags.remove(label)
                    adjusted = True
            if adjusted:
                pct = int(factor * 100)
                reason_parts.append(
                    "{} genre context: borderline toxicity signals reduced by {}%".format(canonical, pct)
                )

        return {
            "genre_probs": probs,
            "genre_flags": flags,
            "genre_adjusted": adjusted,
            "genre_skipped": False,
            "genre_reason": (
                "; ".join(reason_parts) if reason_parts
                else "Genre context '{}' evaluated — no adjustments needed.".format(canonical)
            )
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_GENRE_EVALUATOR_INSTANCE: Optional[GenreContextEvaluator] = None


def get_genre_evaluator() -> GenreContextEvaluator:
    """Return (or lazily create) the shared GenreContextEvaluator singleton."""
    global _GENRE_EVALUATOR_INSTANCE
    if _GENRE_EVALUATOR_INSTANCE is None:
        _GENRE_EVALUATOR_INSTANCE = GenreContextEvaluator()
    return _GENRE_EVALUATOR_INSTANCE

