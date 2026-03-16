"""
NyayaMitra — Query Router Service.

Classifies incoming legal queries into domain, query type, and
jurisdiction using a trained DistilBERT model with keyword-based
fallback for low-confidence predictions.

Classification outputs:
    domain:       criminal | property | family | constitutional |
                  labor | consumer | ip | general
    query_type:   rights | procedure | case_outcome
    jurisdiction: central | state name (auto-detected from query)
    confidence:   0.0 - 1.0

Strategy:
    1. Run DistilBERT model if available
    2. If model confidence < threshold OR model unavailable:
       apply keyword-based rules
    3. Detect jurisdiction from state/city mentions in query
    4. Return highest-confidence classification

Usage:
    from app.services.query_router import get_query_router

    router = await get_query_router()
    result = router.classify("Can police arrest me without a warrant?")
    # {"domain": "criminal", "query_type": "rights",
    #  "jurisdiction": "central", "confidence": 0.94}
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import structlog

from app.config import settings

logger = structlog.get_logger()

# Model path
MODEL_DIR = Path(settings.ROUTER_MODEL_PATH) if hasattr(settings, "ROUTER_MODEL_PATH") else None
if MODEL_DIR and not MODEL_DIR.is_absolute():
    MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / MODEL_DIR


# ═══════════════════════════════════════════════════════════════════════════════
# Keyword Rules (Fallback)
# ═══════════════════════════════════════════════════════════════════════════════

DOMAIN_KEYWORDS = {
    "criminal": [
        "arrest", "fir", "bail", "murder", "theft", "robbery", "rape",
        "assault", "kidnap", "police", "crime", "criminal", "ipc", "crpc",
        "bnss", "bns", "498a", "302", "376", "420", "magistrate",
        "cognizable", "warrant", "custody", "chargesheet", "prosecution",
        "imprisonment", "punishment", "offence", "accused", "victim",
        "dowry", "cheating", "forgery", "defamation", "intimidation",
        "abetment", "conspiracy", "rioting", "dacoity", "ndps", "drug",
        "juvenile", "confession", "evidence", "witness", "hitrun",
        "drunk driving", "cyberstalking", "stalking", "acid attack",
    ],
    "property": [
        "property", "land", "tenant", "landlord", "rent", "lease",
        "sale deed", "registration", "stamp duty", "mutation", "tpa",
        "transfer", "rera", "builder", "flat", "apartment", "possession",
        "eviction", "mortgage", "easement", "adverse possession",
        "encumbrance", "conveyance", "benami", "contract", "agreement",
        "breach", "specific performance", "will", "gift deed", "partition",
        "immovable", "movable", "khata", "noc", "encroachment",
    ],
    "family": [
        "divorce", "marriage", "husband", "wife", "spouse", "custody",
        "maintenance", "alimony", "dowry", "domestic violence", "dv act",
        "hma", "sma", "child", "adoption", "guardianship", "conjugal",
        "separation", "annulment", "talaq", "nikah", "remarriage",
        "matrimonial", "family court", "succession", "inheritance",
        "daughter", "son", "ancestor", "ancestral", "stridhan",
        "inter-caste", "inter-religion", "interfaith", "live-in",
        "protection order", "child support",
    ],
    "constitutional": [
        "fundamental right", "constitution", "article", "pil",
        "writ", "habeas corpus", "mandamus", "certiorari",
        "supreme court", "high court", "rti", "right to information",
        "freedom of speech", "equality", "liberty", "privacy",
        "secularism", "dpsp", "directive principle", "amendment",
        "reservation", "basic structure", "judicial review",
        "election", "president", "governor", "parliament", "legislature",
        "federalism", "emergency", "article 14", "article 19",
        "article 21", "article 32", "article 370", "caa",
        "right to education", "discrimination", "untouchability",
        "scheduled caste", "scheduled tribe", "obc",
    ],
    "labor": [
        "employer", "employee", "worker", "wages", "salary",
        "termination", "retrenchment", "strike", "lockout", "union",
        "posh", "sexual harassment", "workplace", "factory",
        "industrial dispute", "labour", "labor", "pf", "provident fund",
        "esi", "gratuity", "maternity", "overtime", "minimum wage",
        "working hours", "contract worker", "gig worker",
        "shops and establishment", "occupational safety",
        "trade union", "conciliation", "tribunal", "unfair practice",
    ],
    "consumer": [
        "consumer", "defective product", "deficiency", "complaint",
        "refund", "replacement", "compensation", "consumer court",
        "consumer forum", "cpa", "misleading", "advertisement",
        "unfair trade", "product liability", "e-commerce", "online shopping",
        "insurance", "banking", "hospital", "medical negligence",
        "food adulteration", "telecom", "electricity bill",
        "flight delay", "cancelled", "overcharge",
    ],
    "ip": [
        "copyright", "trademark", "patent", "intellectual property",
        "it act", "cyber", "hacking", "data protection", "privacy",
        "online", "internet", "social media", "website", "software",
        "piracy", "digital", "intermediary", "safe harbour",
        "66a", "shreya singhal", "takedown", "domain name",
        "geographical indication", "design", "trade secret",
        "revenge porn", "deepfake", "vpn", "encryption",
        "cert-in", "onnx", "screen recording",
    ],
}

QUERY_TYPE_KEYWORDS = {
    "procedure": [
        "how to", "how do i", "what is the process", "what is the procedure",
        "steps to", "file a", "apply for", "register", "complaint",
        "where to", "where do i", "kaise", "kahan", "process of",
        "what should i do", "what to do", "what can i do",
    ],
    "case_outcome": [
        "what did the court", "what did the supreme court",
        "what is the judgment", "landmark case", "decided",
        "supreme court said", "held that", "case about",
        "judgment about", "ruling on", "verdict",
    ],
    # "rights" is the default if neither procedure nor case_outcome matches
}

# State/city → jurisdiction mapping
JURISDICTION_PATTERNS = {
    "Maharashtra": ["maharashtra", "mumbai", "pune", "nagpur", "thane", "bombay"],
    "Delhi": ["delhi", "new delhi", "ncr"],
    "Karnataka": ["karnataka", "bangalore", "bengaluru", "mysore", "mangalore"],
    "Tamil Nadu": ["tamil nadu", "chennai", "madras", "coimbatore", "madurai"],
    "Uttar Pradesh": ["uttar pradesh", "up ", "lucknow", "noida", "agra", "varanasi", "allahabad", "prayagraj"],
    "West Bengal": ["west bengal", "kolkata", "calcutta"],
    "Kerala": ["kerala", "kochi", "thiruvananthapuram", "ernakulam"],
    "Gujarat": ["gujarat", "ahmedabad", "surat", "vadodara", "rajkot"],
    "Telangana": ["telangana", "hyderabad", "secunderabad"],
    "Andhra Pradesh": ["andhra pradesh", "visakhapatnam", "vijayawada", "amaravati"],
    "Rajasthan": ["rajasthan", "jaipur", "jodhpur", "udaipur"],
    "Punjab": ["punjab", "chandigarh", "ludhiana", "amritsar"],
    "Madhya Pradesh": ["madhya pradesh", "bhopal", "indore", "jabalpur"],
    "Bihar": ["bihar", "patna"],
    "Odisha": ["odisha", "orissa", "bhubaneswar", "cuttack"],
    "Goa": ["goa", "panaji"],
    "Jharkhand": ["jharkhand", "ranchi"],
    "Assam": ["assam", "guwahati", "gauhati"],
    "Chhattisgarh": ["chhattisgarh", "raipur"],
    "Chandigarh": ["chandigarh"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Query Router
# ═══════════════════════════════════════════════════════════════════════════════


class QueryRouter:
    """
    Classifies legal queries into domain, query type, and jurisdiction.

    Uses a trained DistilBERT model when available, with keyword-based
    fallback for low-confidence predictions or when the model is not loaded.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self._model_available = False
        self._device = None

    def load_model(self) -> bool:
        """
        Load the trained DistilBERT router model.

        Returns True if model loaded successfully, False otherwise.
        """
        if self._model_available:
            return True

        if not MODEL_DIR or not (MODEL_DIR / "router_model.pt").exists():
            logger.info("router_model_not_found", path=str(MODEL_DIR))
            return False

        try:
            import torch
            from transformers import DistilBertTokenizer

            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu"
            )

            # Load config
            with open(MODEL_DIR / "router_config.json") as f:
                self.config = json.load(f)

            # Load tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained(str(MODEL_DIR))

            # Build and load model
            from data.training.train_router import build_model

            self.model = build_model()
            self.model.load_state_dict(
                torch.load(MODEL_DIR / "router_model.pt", map_location=self._device)
            )
            self.model.to(self._device)
            self.model.eval()

            self._model_available = True
            logger.info(
                "router_model_loaded",
                device=str(self._device),
                domain_acc=self.config.get("best_domain_acc"),
            )
            return True

        except Exception as e:
            logger.warning("router_model_load_failed", error=str(e))
            return False

    def classify(
        self,
        query: str,
        confidence_threshold: float | None = None,
    ) -> dict:
        """
        Classify a legal query.

        Strategy:
            1. Try DistilBERT model
            2. If confidence < threshold, enhance with keyword rules
            3. Detect jurisdiction from query text

        Args:
            query: The user's legal question.
            confidence_threshold: Minimum confidence to trust model
                (default from config: ROUTER_CONFIDENCE_THRESHOLD).

        Returns:
            dict with domain, query_type, jurisdiction, confidence,
            method ("model", "keyword", "hybrid").
        """
        threshold = confidence_threshold or settings.ROUTER_CONFIDENCE_THRESHOLD
        start = time.time()

        # Step 1: Try model classification
        model_result = None
        if self._model_available:
            model_result = self._classify_with_model(query)

        # Step 2: Keyword classification (always computed)
        keyword_result = self._classify_with_keywords(query)

        # Step 3: Merge results
        if model_result and model_result["confidence"] >= threshold:
            # Model is confident — use it, but check if keywords disagree
            final = model_result.copy()
            final["method"] = "model"

            # If keywords strongly disagree, lower confidence
            if (keyword_result["domain"] != "general" and
                    keyword_result["domain"] != model_result["domain"] and
                    keyword_result["confidence"] > 0.7):
                # Keyword and model disagree — use model but flag
                final["keyword_suggestion"] = keyword_result["domain"]
        elif model_result:
            # Model has low confidence — combine with keywords
            if keyword_result["domain"] != "general":
                final = keyword_result.copy()
                final["confidence"] = max(keyword_result["confidence"],
                                          model_result["confidence"])
                final["method"] = "hybrid"
            else:
                final = model_result.copy()
                final["method"] = "model_low_conf"
        else:
            # No model — pure keyword
            final = keyword_result.copy()
            final["method"] = "keyword"

        # Step 4: Detect jurisdiction
        final["jurisdiction"] = self._detect_jurisdiction(query)

        # Step 5: Override query_type with keyword detection (more reliable)
        keyword_qtype = self._detect_query_type(query)
        if keyword_qtype != "rights":  # rights is default, only override if specific match
            final["query_type"] = keyword_qtype

        duration_ms = round((time.time() - start) * 1000, 2)
        final["duration_ms"] = duration_ms

        logger.info(
            "query_classified",
            domain=final["domain"],
            query_type=final["query_type"],
            jurisdiction=final["jurisdiction"],
            confidence=round(final["confidence"], 3),
            method=final["method"],
            duration_ms=duration_ms,
        )

        return final

    # ─── Model-based Classification ──────────────────────────────────────

    def _classify_with_model(self, query: str) -> dict | None:
        """Run the DistilBERT model on the query."""
        try:
            import torch

            inputs = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=self.config.get("max_length", 128),
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].to(self._device)
            attention_mask = inputs["attention_mask"].to(self._device)

            with torch.no_grad():
                domain_logits, qtype_logits = self.model(input_ids, attention_mask)

            # Domain prediction
            domain_probs = torch.softmax(domain_logits, dim=1).squeeze()
            domain_id = domain_probs.argmax().item()
            domain_conf = domain_probs[domain_id].item()
            domain_labels = self.config.get("domain_labels", [])
            domain = domain_labels[domain_id] if domain_id < len(domain_labels) else "general"

            # Query type prediction
            qtype_probs = torch.softmax(qtype_logits, dim=1).squeeze()
            qtype_id = qtype_probs.argmax().item()
            qtype_labels = self.config.get("qtype_labels", [])
            query_type = qtype_labels[qtype_id] if qtype_id < len(qtype_labels) else "rights"

            return {
                "domain": domain,
                "query_type": query_type,
                "confidence": domain_conf,
                "domain_probs": {
                    label: round(prob.item(), 4)
                    for label, prob in zip(domain_labels, domain_probs)
                },
            }

        except Exception as e:
            logger.warning("model_classify_error", error=str(e))
            return None

    # ─── Keyword-based Classification ────────────────────────────────────

    def _classify_with_keywords(self, query: str) -> dict:
        """Classify using keyword matching."""
        query_lower = query.lower()

        # Score each domain
        scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw in query_lower:
                    # Longer keywords get higher weight
                    weight = 1.0 + (len(kw.split()) - 1) * 0.5
                    score += weight
            scores[domain] = score

        # Find best domain
        best_domain = max(scores, key=scores.get) if max(scores.values()) > 0 else "general"
        total_score = sum(scores.values())
        confidence = scores[best_domain] / total_score if total_score > 0 else 0.0

        # Normalize confidence to 0-1 range (cap at 0.9 for keyword)
        confidence = min(confidence, 0.9)

        # Query type
        query_type = self._detect_query_type(query)

        return {
            "domain": best_domain,
            "query_type": query_type,
            "confidence": confidence,
        }

    def _detect_query_type(self, query: str) -> str:
        """Detect query type using keyword patterns."""
        query_lower = query.lower()

        for qtype, patterns in QUERY_TYPE_KEYWORDS.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return qtype

        return "rights"  # default

    def _detect_jurisdiction(self, query: str) -> str:
        """Detect jurisdiction from state/city mentions."""
        query_lower = query.lower()

        for state, patterns in JURISDICTION_PATTERNS.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return state

        return "central"


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_query_router: QueryRouter | None = None


async def get_query_router() -> QueryRouter:
    """Get or create the singleton query router."""
    global _query_router
    if _query_router is None:
        _query_router = QueryRouter()
        _query_router.load_model()  # non-fatal if model not found
    return _query_router