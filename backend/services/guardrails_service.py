"""
Guardrails Service
==================
Input and output safety checks for the Agentic RAG system.

Input Guardrails:
  - Prompt injection detection
  - Topic restriction (domain guard)
  - Input length validation

Output Guardrails:
  - PII detection and redaction
  - Hallucination check (groundedness)
  - Response relevance check
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import structlog

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Domain keywords — queries must relate to at least one of these areas
# ---------------------------------------------------------------------------
DOMAIN_KEYWORDS = [
    # Procurement
    "procurement", "purchase", "vendor", "supplier", "tender", "rfp", "rfq",
    "contract", "bid", "bidding", "sourcing", "requisition", "invoice",
    "payment", "budget", "spending", "cost", "price", "quotation",
    "purchase order", "po", "approval", "threshold", "evaluation",
    # HR
    "hr", "human resource", "employee", "leave", "salary", "policy",
    "bylaw", "bylaws", "termination", "hiring", "onboarding", "benefits",
    "probation", "resignation", "disciplinary", "grievance", "attendance",
    # Information Security
    "security", "information security", "access control", "password",
    "data protection", "classification", "confidential", "encryption",
    "incident", "compliance", "audit", "risk",
    # Document-related
    "document", "manual", "standard", "guideline", "regulation",
    "abu dhabi", "adgm", "procedure", "process", "rule", "rules",
]

# ---------------------------------------------------------------------------
# Prompt injection patterns
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(your|previous|above|prior)\s+(instructions|rules|prompts)",
    r"forget\s+(everything|all|your)\s+(above|previous|instructions|rules)",
    r"you\s+are\s+now\s+(a|an|the)\s+",
    r"new\s+(system\s+)?prompt\s*:",
    r"system\s*:\s*you\s+are",
    r"act\s+as\s+(a|an|if)\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"override\s+(your|all|the)\s+(instructions|rules|settings)",
    r"disregard\s+(your|all|the|previous)\s+(instructions|rules)",
    r"do\s+not\s+follow\s+(your|the)\s+(instructions|rules|guidelines)",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"ignore\s+safety",
    r"bypass\s+(filter|safety|guard|restriction)",
    r"reveal\s+(your|the)\s+(system|initial|original)\s+prompt",
    r"what\s+is\s+your\s+system\s+prompt",
    r"repeat\s+(your|the)\s+(instructions|system\s+prompt|rules)\s+(back|above)",
]


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    passed: bool
    guardrail_name: str
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailStats:
    """Track guardrail activity."""
    total_input_checks: int = 0
    total_output_checks: int = 0
    input_blocks: int = 0
    output_modifications: int = 0
    injection_detections: int = 0
    off_topic_detections: int = 0
    pii_redactions: int = 0
    hallucination_warnings: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_input_checks": self.total_input_checks,
            "total_output_checks": self.total_output_checks,
            "input_blocks": self.input_blocks,
            "output_modifications": self.output_modifications,
            "injection_detections": self.injection_detections,
            "off_topic_detections": self.off_topic_detections,
            "pii_redactions": self.pii_redactions,
            "hallucination_warnings": self.hallucination_warnings,
        }


class GuardrailsService:
    """
    Input and output guardrails for the RAG system.

    Usage:
        guardrails = GuardrailsService()

        # Before LLM call
        input_result = guardrails.check_input(user_query)
        if not input_result.passed:
            return input_result.message  # Block the request

        # After LLM call
        output_result = guardrails.check_output(response, sources, user_query)
        final_response = output_result.details.get("cleaned_response", response)
    """

    def __init__(
        self,
        enabled: bool = True,
        enable_injection_detection: bool = True,
        enable_topic_restriction: bool = True,
        enable_input_length_check: bool = True,
        enable_pii_redaction: bool = True,
        enable_hallucination_check: bool = True,
        enable_relevance_check: bool = True,
        min_query_length: int = 3,
        max_query_length: int = 2000,
        hallucination_threshold: float = 0.1,
    ):
        self.enabled = enabled
        self.enable_injection_detection = enable_injection_detection
        self.enable_topic_restriction = enable_topic_restriction
        self.enable_input_length_check = enable_input_length_check
        self.enable_pii_redaction = enable_pii_redaction
        self.enable_hallucination_check = enable_hallucination_check
        self.enable_relevance_check = enable_relevance_check
        self.min_query_length = min_query_length
        self.max_query_length = max_query_length
        self.hallucination_threshold = hallucination_threshold
        self.stats = GuardrailStats()

        # Compile regex patterns once for performance
        self._injection_patterns = [
            re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
        ]
        self._email_pattern = re.compile(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        )
        self._phone_pattern = re.compile(
            r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
        )

    # -----------------------------------------------------------------------
    # INPUT GUARDRAILS
    # -----------------------------------------------------------------------

    def check_input(self, query: str) -> GuardrailResult:
        """
        Run all input guardrails on the user query.

        Returns GuardrailResult with passed=False if any check fails.
        """
        if not self.enabled:
            return GuardrailResult(passed=True, guardrail_name="disabled")

        self.stats.total_input_checks += 1

        # 1. Input length check
        if self.enable_input_length_check:
            result = self._check_input_length(query)
            if not result.passed:
                self.stats.input_blocks += 1
                logger.warning("Input blocked by length check",
                             query_length=len(query), reason=result.message)
                return result

        # 2. Prompt injection detection
        if self.enable_injection_detection:
            result = self._check_prompt_injection(query)
            if not result.passed:
                self.stats.input_blocks += 1
                self.stats.injection_detections += 1
                logger.warning("Input blocked by injection detection",
                             query=query[:100], reason=result.message)
                return result

        # 3. Topic restriction
        if self.enable_topic_restriction:
            result = self._check_topic_restriction(query)
            if not result.passed:
                self.stats.input_blocks += 1
                self.stats.off_topic_detections += 1
                logger.warning("Input blocked by topic restriction",
                             query=query[:100], reason=result.message)
                return result

        return GuardrailResult(passed=True, guardrail_name="all_input_checks")

    def _check_input_length(self, query: str) -> GuardrailResult:
        """Check if query length is within acceptable bounds."""
        query_stripped = query.strip()
        length = len(query_stripped)

        if length < self.min_query_length:
            return GuardrailResult(
                passed=False,
                guardrail_name="input_length",
                message="Your query is too short. Please provide a more detailed question.",
                details={"length": length, "min_required": self.min_query_length},
            )

        if length > self.max_query_length:
            return GuardrailResult(
                passed=False,
                guardrail_name="input_length",
                message=f"Your query is too long ({length} characters). Please keep it under {self.max_query_length} characters.",
                details={"length": length, "max_allowed": self.max_query_length},
            )

        return GuardrailResult(passed=True, guardrail_name="input_length")

    def _check_prompt_injection(self, query: str) -> GuardrailResult:
        """Detect prompt injection attempts."""
        query_lower = query.lower()

        for pattern in self._injection_patterns:
            match = pattern.search(query_lower)
            if match:
                return GuardrailResult(
                    passed=False,
                    guardrail_name="prompt_injection",
                    message="I can only help with questions about the documents in my knowledge base. Please ask a question related to procurement, HR policies, or information security.",
                    details={"matched_pattern": match.group()},
                )

        return GuardrailResult(passed=True, guardrail_name="prompt_injection")

    def _check_topic_restriction(self, query: str) -> GuardrailResult:
        """Check if the query is related to the allowed domain."""
        query_lower = query.lower()

        # Check if any domain keyword appears in the query (word boundary match)
        for keyword in DOMAIN_KEYWORDS:
            if len(keyword) <= 3:
                # Short keywords need word boundary matching to avoid false positives
                if re.search(rf"\b{re.escape(keyword)}\b", query_lower):
                    return GuardrailResult(passed=True, guardrail_name="topic_restriction")
            elif keyword in query_lower:
                return GuardrailResult(passed=True, guardrail_name="topic_restriction")

        return GuardrailResult(
            passed=False,
            guardrail_name="topic_restriction",
            message="I'm specialized in procurement, HR policies, and information security documents. Your question seems outside my area of expertise. Please ask something related to these topics.",
            details={"query": query[:100]},
        )

    # -----------------------------------------------------------------------
    # OUTPUT GUARDRAILS
    # -----------------------------------------------------------------------

    def check_output(
        self,
        response: str,
        sources: List[Dict[str, Any]],
        original_query: str,
    ) -> GuardrailResult:
        """
        Run all output guardrails on the LLM response.

        Unlike input guardrails that BLOCK, output guardrails MODIFY
        the response (e.g., redact PII) and add warnings.

        Returns GuardrailResult with:
          - details["cleaned_response"]: the potentially modified response
          - details["warnings"]: list of warning messages
        """
        if not self.enabled:
            return GuardrailResult(
                passed=True,
                guardrail_name="disabled",
                details={"cleaned_response": response, "warnings": []},
            )

        self.stats.total_output_checks += 1
        cleaned_response = response
        warnings: List[str] = []

        # 1. PII redaction
        if self.enable_pii_redaction:
            cleaned_response, pii_found = self._redact_pii(cleaned_response)
            if pii_found:
                self.stats.output_modifications += 1
                self.stats.pii_redactions += 1
                warnings.append("PII detected and redacted from response.")
                logger.info("PII redacted from response", pii_types=pii_found)

        # 2. Hallucination check
        if self.enable_hallucination_check and sources:
            hallucination_result = self._check_hallucination(cleaned_response, sources)
            if not hallucination_result.passed:
                self.stats.hallucination_warnings += 1
                warnings.append(hallucination_result.message)
                logger.warning("Potential hallucination detected",
                             overlap_score=hallucination_result.details.get("overlap_score"))

        # 3. Relevance check
        if self.enable_relevance_check:
            relevance_result = self._check_relevance(cleaned_response, original_query)
            if not relevance_result.passed:
                warnings.append(relevance_result.message)
                logger.warning("Low relevance detected",
                             overlap_score=relevance_result.details.get("overlap_score"))

        return GuardrailResult(
            passed=True,
            guardrail_name="all_output_checks",
            details={
                "cleaned_response": cleaned_response,
                "warnings": warnings,
            },
        )

    def _redact_pii(self, text: str) -> tuple[str, List[str]]:
        """Detect and redact PII from text. Returns (cleaned_text, list_of_pii_types_found)."""
        pii_found = []

        # Redact emails
        if self._email_pattern.search(text):
            text = self._email_pattern.sub("[EMAIL_REDACTED]", text)
            pii_found.append("email")

        # Redact phone numbers
        if self._phone_pattern.search(text):
            text = self._phone_pattern.sub("[PHONE_REDACTED]", text)
            pii_found.append("phone")

        return text, pii_found

    def _check_hallucination(
        self, response: str, sources: List[Dict[str, Any]]
    ) -> GuardrailResult:
        """
        Check if the response is grounded in the source documents.

        Compares key terms in the response against the source content.
        Low overlap suggests the LLM may be hallucinating.
        """
        # Combine all source text
        source_text = " ".join(
            s.get("content", "") + " " + s.get("text", "")
            for s in sources
        ).lower()

        if not source_text.strip():
            return GuardrailResult(passed=True, guardrail_name="hallucination_check")

        # Extract meaningful words from response (skip common words)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "and", "but", "or",
            "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
            "every", "all", "any", "few", "more", "most", "some", "such", "than",
            "too", "very", "just", "also", "that", "this", "these", "those",
            "it", "its", "they", "them", "their", "we", "our", "you", "your",
            "i", "me", "my", "he", "she", "his", "her", "which", "what", "who",
            "whom", "how", "when", "where", "why", "if", "then", "else",
        }

        response_words = set(re.findall(r"[a-zA-Z]{3,}", response.lower()))
        meaningful_words = response_words - stop_words

        if not meaningful_words:
            return GuardrailResult(passed=True, guardrail_name="hallucination_check")

        # Count how many response words appear in source documents
        grounded_words = sum(1 for w in meaningful_words if w in source_text)
        overlap_score = grounded_words / len(meaningful_words)

        if overlap_score < self.hallucination_threshold:
            return GuardrailResult(
                passed=False,
                guardrail_name="hallucination_check",
                message=f"Warning: This response may contain information not found in the source documents (groundedness score: {overlap_score:.0%}).",
                details={"overlap_score": overlap_score, "grounded_words": grounded_words, "total_words": len(meaningful_words)},
            )

        return GuardrailResult(
            passed=True,
            guardrail_name="hallucination_check",
            details={"overlap_score": overlap_score},
        )

    def _check_relevance(self, response: str, query: str) -> GuardrailResult:
        """Check if the response is relevant to the original query."""
        query_words = set(re.findall(r"[a-zA-Z]{4,}", query.lower()))
        response_words = set(re.findall(r"[a-zA-Z]{4,}", response.lower()))

        if not query_words or not response_words:
            return GuardrailResult(passed=True, guardrail_name="relevance_check")

        # Check how many query words appear in the response
        overlap = len(query_words & response_words)
        overlap_score = overlap / len(query_words)

        if overlap_score < 0.1:
            return GuardrailResult(
                passed=False,
                guardrail_name="relevance_check",
                message="Warning: The response may not directly address your question.",
                details={"overlap_score": overlap_score},
            )

        return GuardrailResult(
            passed=True,
            guardrail_name="relevance_check",
            details={"overlap_score": overlap_score},
        )

    # -----------------------------------------------------------------------
    # CONFIGURATION & STATS
    # -----------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.to_dict()

    def get_config(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "input_guardrails": {
                "injection_detection": self.enable_injection_detection,
                "topic_restriction": self.enable_topic_restriction,
                "input_length_check": self.enable_input_length_check,
                "min_query_length": self.min_query_length,
                "max_query_length": self.max_query_length,
            },
            "output_guardrails": {
                "pii_redaction": self.enable_pii_redaction,
                "hallucination_check": self.enable_hallucination_check,
                "relevance_check": self.enable_relevance_check,
                "hallucination_threshold": self.hallucination_threshold,
            },
        }

    def update_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info("Guardrail config updated", key=key, value=value)
