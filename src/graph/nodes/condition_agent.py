"""
Condition Agent Node
LLM 기반 지능형 라우팅 에이전트
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Literal
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, asdict

import httpx
from langchain_core.messages import HumanMessage
from supabase import create_client, Client

from src.graph.nodes.base import Node
from src.models.do import RawResponse
from src.utils.logger import setup_logger


# ===== Configuration =====
class Config:
    """전역 설정 상수"""
    MAX_DATA_SOURCES = 10
    MAX_RETRIES = 2
    DEFAULT_CONFIDENCE = 0.8
    TIMEOUT_SECONDS = 30

    # LLM 설정
    DEFAULT_MODEL = "openai/gpt-4o-mini"
    TEMPERATURE = 0.1
    MAX_TOKENS = 2500
    TOP_P = 0.95

    # API 설정
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


# ===== Custom Exceptions =====
class RoutingError(Exception):
    """라우팅 관련 에러"""
    pass


class LLMError(Exception):
    """LLM 호출 관련 에러"""
    pass


class ProfileError(Exception):
    """프로필 조회 관련 에러"""
    pass


# ===== Enums and Data Classes =====
class RouteType(Enum):
    """라우팅 타입 정의"""
    DEBATE = "debate_block"
    GUARDRAIL = "guardrail_layer"
    RETRIEVAL = "retrieval_pipeline"
    FINANCE = "finance_agent"
    ONBOARDING = "onboarding"


@dataclass
class InvestmentAdvisorySignals:
    """
    투자자문 의도 분석 결과

    Attributes:
        is_seeking_recommendation: 투자 추천 요청 여부
        is_timing_advice: 매매 시점 조언 요청 여부
        is_portfolio_allocation: 포트폴리오 배분 조언 여부
        personalization_level: 개인화 수준 (HIGH: 개인맞춤, MEDIUM: 일반적, LOW: 범용)
        risk_level: 위험도 수준 (HIGH: 고위험, MEDIUM: 중위험, LOW: 저위험)
    """
    is_seeking_recommendation: bool
    is_timing_advice: bool
    is_portfolio_allocation: bool
    personalization_level: Literal["HIGH", "MEDIUM", "LOW"]
    risk_level: Literal["HIGH", "MEDIUM", "LOW"]


@dataclass
class IntentAnalysis:
    """
    사용자 의도 분석 결과

    Attributes:
        primary_intent: 주요 의도
        secondary_intents: 부가 의도 목록
        entities: 추출된 엔티티 (회사명, 상품명, 금액 등)
        complexity_level: 복잡도 수준
        requires_comparison: 비교 분석 필요 여부
        requires_realtime_data: 실시간 데이터 필요 여부
        requires_historical_analysis: 과거 데이터 분석 필요 여부
        sentiment: 감정 상태 (positive/negative/neutral/questioning)
    """
    primary_intent: str
    secondary_intents: List[str]
    entities: Dict[str, List[str]]
    complexity_level: Literal["HIGH", "MEDIUM", "LOW"]
    requires_comparison: bool
    requires_realtime_data: bool
    requires_historical_analysis: bool
    sentiment: Literal["positive", "negative", "neutral", "questioning"]


# ===== LLM Client =====
class LLMClient:
    """OpenAI Compatible API 클라이언트"""

    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url or Config.DEFAULT_BASE_URL
        self.logger = setup_logger(self.__class__.__name__)

    async def complete(
            self,
            system_prompt: str,
            user_prompt: str,
            model: str = None,
            temperature: float = None,
            max_tokens: int = None
    ) -> Dict:
        """
        LLM 완성 요청

        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            model: 사용할 모델 (기본: Config.DEFAULT_MODEL)
            temperature: 샘플링 온도 (기본: Config.TEMPERATURE)
            max_tokens: 최대 토큰 수 (기본: Config.MAX_TOKENS)

        Returns:
            파싱된 JSON 응답

        Raises:
            LLMError: API 호출 실패 시
        """
        model = model or os.getenv("MAIN_LLM_MODEL", Config.DEFAULT_MODEL)
        temperature = temperature or Config.TEMPERATURE
        max_tokens = max_tokens or Config.MAX_TOKENS

        if not self.api_key:
            raise LLMError("API key not found")

        retry_count = 0

        while retry_count < Config.MAX_RETRIES:
            try:
                async with httpx.AsyncClient(timeout=Config.TIMEOUT_SECONDS) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "top_p": Config.TOP_P,
                        },
                    )

                    if response.status_code == 200:
                        result = response.json()
                        content = result["choices"][0]["message"]["content"]
                        return self._parse_json_response(content)

                    elif response.status_code == 429:  # Rate limit
                        retry_count += 1
                        await asyncio.sleep(2 ** retry_count)
                        continue

                    else:
                        self.logger.error(f"API Error: {response.status_code}")
                        retry_count += 1

            except asyncio.TimeoutError:
                self.logger.error("API timeout")
                retry_count += 1

            except Exception as e:
                self.logger.error(f"Error calling API: {e}")
                retry_count += 1

        raise LLMError(f"Failed after {Config.MAX_RETRIES} retries")

    def _parse_json_response(self, content: str) -> Dict:
        """JSON 응답 파싱"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # JSON 추출 시도
            cleaned = self._extract_json_from_text(content)
            return json.loads(cleaned)

    def _extract_json_from_text(self, text: str) -> str:
        """텍스트에서 JSON 추출"""
        import re

        patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        match = match[0]
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return match
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

        # 최후의 시도
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1:
            potential_json = text[first_brace:last_brace + 1]
            try:
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass

        raise LLMError("Could not extract JSON from response")


# ===== Main Agent Node =====
class ConditionAgentNode(Node):
    """
    LLM 지능형 라우팅 에이전트 노드
    사용자 요청을 분석하여 적절한 처리 경로로 라우팅
    """

    # 라우트 설명 매핑
    ROUTE_DESCRIPTIONS = {
        RouteType.DEBATE: "For comparison analysis, decision-making, multiple perspectives needed",
        RouteType.GUARDRAIL: "When regulatory compliance check is needed (NOT for blocking investment advice)",
        RouteType.RETRIEVAL: "For general information queries and knowledge retrieval",
        RouteType.FINANCE: "For real-time market data, stock prices, financial metrics",
        RouteType.ONBOARDING: "For new user registration and profile setup"
    }

    def __init__(self):
        super().__init__()
        self.logger = setup_logger(self.__class__.__name__)

        # LLM 클라이언트 초기화
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", Config.DEFAULT_BASE_URL)
        self.llm_client = LLMClient(api_key, base_url)

        # Supabase 클라이언트 초기화
        self.supabase_url = os.getenv("SUPABASE_URL", "")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY", "")

        if self.supabase_url and self.supabase_key:
            self.supabase: Optional[Client] = create_client(
                self.supabase_url, self.supabase_key
            )
        else:
            self.supabase = None
            self.logger.warning("Supabase credentials not found. Running in offline mode.")

    def _generate_routing_prompt(self) -> str:
        """동적 라우팅 프롬프트 생성"""
        routes_description = []
        for i, route_type in enumerate(RouteType, 1):
            description = self.ROUTE_DESCRIPTIONS.get(route_type)
            routes_description.append(f"{i}. {route_type.value}: {description}")

        return f"""You are an intelligent routing agent for a financial AI system.
        Your role is to analyze user requests comprehensively and determine the most appropriate processing path.
        
        Consider all aspects:
        - User intent and context
        - Complexity of the request
        - Regulatory implications
        - Required data sources
        - Edge cases and exceptions
        
        Available routes:
        {chr(10).join(routes_description)}
        
        Be flexible and context-aware in your routing decisions."""

    def _run(self, state: dict) -> dict:
        """Supervisor 노드에서 호출되는 메인 실행 함수"""
        try:
            messages = state.get("messages", [])
            if not messages:
                raise RoutingError("No messages in state")

            last_message = messages[-1]
            user_input = (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message)
            )

            user_id = state.get("user_id", "default_user")

            # 비동기 함수 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.process_request(user_input, user_id)
                )
            finally:
                loop.close()

            response_content = self._format_routing_result(result)
            self.logger.info(f"Routing decision: {result['route']}")

            return {
                "messages": [
                    HumanMessage(content=response_content, name="condition_agent")
                ],
                "routing_result": result,
            }

        except Exception as e:
            self.logger.error(f"Error in _run: {str(e)}")
            return {
                "messages": [
                    HumanMessage(
                        content=f"라우팅 처리 중 오류 발생: {str(e)}",
                        name="condition_agent",
                    )
                ]
            }

    def _invoke(self, query: str) -> RawResponse:
        """API 엔드포인트용 함수"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.process_request(query, "api_user")
                )
            finally:
                loop.close()

            return RawResponse(answer=json.dumps(result, ensure_ascii=False, indent=2))

        except Exception as e:
            self.logger.error(f"Error in _invoke: {str(e)}")
            return RawResponse(answer=f"Error: {str(e)}")

    async def process_request(self, user_input: str, user_id: str) -> Dict[str, Any]:
        """
        메인 라우팅 처리 파이프라인

        Args:
            user_input: 사용자 입력
            user_id: 사용자 ID

        Returns:
            라우팅 결과 딕셔너리
        """
        # 1. 사용자 프로필 조회
        user_profile = await self._get_user_profile_from_db(user_id)

        # 2. LLM 기반 종합 분석
        routing_result = await self._comprehensive_llm_analysis(
            user_input, user_profile
        )

        # 3. 투자자문 의도 정밀 분석 (필요시)
        if routing_result.get("needs_compliance_check", False):
            compliance_analysis = await self._analyze_investment_advisory_intent(
                user_input, user_profile, routing_result
            )
            routing_result["compliance_analysis"] = compliance_analysis

        # 4. 동적 데이터 소스 선택
        data_sources = await self._dynamic_data_source_selection(
            user_input, routing_result, user_profile
        )

        # 5. 최종 결과 구성
        result = {
            "route": routing_result["route"],
            "routing_reasoning": routing_result["reasoning"],
            "intent_analysis": routing_result["intent_analysis"],
            "compliance_status": routing_result.get("compliance_analysis", {}),
            "user_profile": user_profile,
            "data_sources": data_sources,
            "confidence_scores": routing_result.get("confidence_scores", {}),
            "context": {
                "user_input": user_input,
                "timestamp": datetime.now().isoformat(),
                "overall_confidence": routing_result.get(
                    "overall_confidence", Config.DEFAULT_CONFIDENCE
                ),
            },
        }

        # 6. 비동기 로깅
        if self.supabase:
            asyncio.create_task(self._save_interaction_log(user_id, result))

        return result

    async def _get_user_profile_from_db(self, user_id: str) -> Optional[Dict]:
        """
        Supabase DB에서 사용자 프로필 조회

        Args:
            user_id: 사용자 ID

        Returns:
            사용자 프로필 딕셔너리 또는 None
        """
        if not self.supabase:
            return {"user_id": user_id, "status": "offline_mode"}

        try:
            response = (
                self.supabase.table("user_profile")
                .select("*")
                .eq("external_user_key", user_id)
                .single()
                .execute()
            )

            if response.data:
                return response.data

            return None  # 온보딩 필요

        except Exception as e:
            self.logger.error(f"Error fetching user profile: {e}")
            raise ProfileError(f"Failed to fetch profile for user {user_id}")

    async def _comprehensive_llm_analysis(
            self, user_input: str, user_profile: Optional[Dict]
    ) -> Dict:
        """
        LLM을 사용한 종합 분석 - 라우팅, 의도분석, 규제체크 통합

        Returns:
            분석 결과 딕셔너리
        """
        system_prompt = f"""You are a comprehensive financial AI routing expert.
        Analyze the user request holistically and make intelligent routing decisions.
        
        Your analysis should consider:
        1. Primary intent and secondary intents
        2. Complexity and risk level
        3. Regulatory implications (Korean financial regulations)
        4. Required expertise and data sources
        5. Edge cases and exceptions
        
        Routing options:
        {self._get_routes_for_prompt()}
        
        For investment advisory detection:
        - Analyze if the request seeks personalized investment recommendations
        - Consider user's experience level and risk profile
        - Check for timing advice, portfolio allocation, specific buy/sell signals
        
        Be adaptive and context-aware. Handle edge cases intelligently.
        
        Respond in structured JSON format."""

        profile_info = (
            self._format_profile_for_prompt(user_profile)
            if user_profile
            else "New user - no profile available"
        )

        user_prompt = f"""
User Profile:
{profile_info}

User Request: "{user_input}"

Provide comprehensive analysis with exact structure:
{{
    "route": "one of: {', '.join([rt.value for rt in RouteType])}",
    "overall_confidence": 0.95,
    "reasoning": "detailed reasoning for routing decision",
    "intent_analysis": {{
        "primary_intent": "main user intent",
        "secondary_intents": ["additional intents"],
        "entities": {{
            "companies": ["mentioned companies"],
            "products": ["financial products"],
            "concepts": ["financial concepts"],
            "time_frames": ["mentioned time periods"],
            "amounts": ["monetary amounts"]
        }},
        "complexity_level": "HIGH/MEDIUM/LOW",
        "requires_comparison": true/false,
        "requires_realtime_data": true/false,
        "requires_historical_analysis": true/false,
        "sentiment": "positive/negative/neutral/questioning"
    }},
    "needs_compliance_check": true/false,
    "investment_advisory_signals": {{
        "is_seeking_recommendation": true/false,
        "is_timing_advice": true/false,
        "is_portfolio_allocation": true/false,
        "personalization_level": "HIGH/MEDIUM/LOW",
        "risk_level": "HIGH/MEDIUM/LOW"
    }},
    "confidence_scores": {{
        "routing_confidence": 0.95,
        "intent_confidence": 0.90,
        "compliance_confidence": 0.85
    }},
    "edge_case_handling": "any special considerations"
}}
"""

        try:
            result = await self.llm_client.complete(system_prompt, user_prompt)
            return self._validate_and_enhance_response(result)
        except LLMError:
            return self._get_intelligent_fallback(user_input)

    async def _analyze_investment_advisory_intent(
            self, user_input: str, user_profile: Dict, routing_result: Dict
    ) -> Dict:
        """투자자문 의도 정밀 분석"""
        system_prompt = """You are a financial regulatory compliance expert specializing in investment advisory detection.
        
        Analyze if the user request constitutes investment advisory that requires licensed professionals.
        
        Investment Advisory Indicators:
        - Personalized buy/sell recommendations
        - Specific timing advice for trades
        - Portfolio allocation suggestions
        - Risk-return optimization advice
        - Individual financial planning
        
        NOT Investment Advisory:
        - General market information
        - Educational content
        - Historical data analysis
        - Product feature comparisons
        - News and public information
        
        Consider the user's profile and sophistication level.
        Professional investors may have different needs than retail investors.
        
        Provide nuanced analysis, not binary blocking."""

        user_prompt = f"""
User Profile Summary:
- Investment Experience: {user_profile.get("invest_experience_yr", "Unknown")} years
- Risk Tolerance: {user_profile.get("risk_tolerance_level", "Unknown")}
- Financial Knowledge: {user_profile.get("financial_knowledge_level", "Unknown")}

User Request: "{user_input}"

Initial Analysis: {json.dumps(routing_result.get("investment_advisory_signals", {}), ensure_ascii=False)}

Provide detailed compliance analysis:
{{
    "is_investment_advisory": true/false,
    "confidence": 0.95,
    "advisory_type": "PERSONAL_RECOMMENDATION/TIMING_ADVICE/PORTFOLIO_MGMT/GENERAL_INFO/EDUCATIONAL",
    "regulatory_risk": "HIGH/MEDIUM/LOW/NONE",
    "required_disclaimers": ["necessary disclaimers if any"],
    "alternative_response_strategy": "how to handle this appropriately",
    "detailed_reasoning": "comprehensive explanation",
    "can_proceed_with_disclaimers": true/false
}}
"""

        return await self.llm_client.complete(system_prompt, user_prompt)

    async def _dynamic_data_source_selection(
            self, user_input: str, routing_result: Dict, user_profile: Dict
    ) -> List[str]:
        """LLM 기반 동적 데이터 소스 선택"""
        system_prompt = """You are a data architecture expert for a financial AI system.
        
        Select optimal data sources based on the request context and routing decision.
        
        Available data sources:
        - user_profile: User demographics, preferences, risk profile
        - document_store: Knowledge base documents, reports
        - document_vector_index: Semantic search embeddings
        - product_metadata_db: Financial product specifications
        - product_metrics: Performance metrics, returns, risks
        - product_embedding_index: Product similarity search
        - market_data_api: Real-time prices, volumes
        - news_feed: Latest financial news
        - regulatory_db: Compliance rules, guidelines
        - historical_prices: Time series data
        - company_fundamentals: Financial statements, ratios
        
        Consider:
        - Query requirements
        - User sophistication level
        - Response latency needs
        - Data freshness requirements
        
        Select minimal but sufficient sources."""

        user_prompt = f"""
Routing Decision: {routing_result["route"]}
User Query: "{user_input}"
Intent Analysis: {json.dumps(routing_result["intent_analysis"], ensure_ascii=False)}
User Profile Level: {user_profile.get("financial_knowledge_level", "Unknown")}

Select data sources:
{{
    "required_sources": ["essential data sources"],
    "optional_sources": ["nice-to-have sources"],
    "excluded_sources": ["explicitly not needed"],
    "source_priority": {{
        "source_name": "priority_level (1-5)"
    }},
    "data_strategy": "how to use these sources",
    "fallback_sources": ["backup if primary fails"],
    "estimated_latency": "expected response time",
    "reasoning": "selection rationale"
}}
"""

        result = await self.llm_client.complete(system_prompt, user_prompt)

        required = result.get("required_sources", [])
        optional = result.get("optional_sources", [])

        all_sources = required + [s for s in optional if s not in required]
        return all_sources[:Config.MAX_DATA_SOURCES]

    def _get_routes_for_prompt(self) -> str:
        """프롬프트용 라우트 설명 생성"""
        lines = []
        for route_type in RouteType:
            desc = self.ROUTE_DESCRIPTIONS.get(route_type)
            lines.append(f"- {route_type.value}: {desc}")
        return "\n".join(lines)

    def _validate_and_enhance_response(self, response: Dict) -> Dict:
        """LLM 응답 검증 및 보강"""
        required_fields = ["route", "reasoning", "intent_analysis"]

        if not all(field in response for field in required_fields):
            # 필수 필드 보강
            return {
                "route": response.get("route", RouteType.RETRIEVAL.value),
                "reasoning": response.get(
                    "reasoning", "Automated routing based on partial analysis"
                ),
                "intent_analysis": response.get(
                    "intent_analysis",
                    {
                        "primary_intent": "information_request",
                        "complexity_level": "MEDIUM"
                    },
                ),
                "overall_confidence": response.get(
                    "overall_confidence", Config.DEFAULT_CONFIDENCE * 0.75
                ),
                **response  # 나머지 필드 유지
            }

        return response

    def _get_intelligent_fallback(self, user_prompt: str) -> Dict:
        """지능형 폴백 - 휴리스틱 기반"""
        route = RouteType.RETRIEVAL.value

        comparison_keywords = ["비교", "vs", "versus", "차이", "어느", "어떤", "선택"]
        realtime_keywords = ["현재", "지금", "실시간", "가격", "시세", "주가"]
        risk_keywords = ["위험", "리스크", "규제", "법적", "컴플라이언스"]

        lower_prompt = user_prompt.lower()

        if any(keyword in lower_prompt for keyword in comparison_keywords):
            route = RouteType.DEBATE.value
        elif any(keyword in lower_prompt for keyword in realtime_keywords):
            route = RouteType.FINANCE.value
        elif any(keyword in lower_prompt for keyword in risk_keywords):
            route = RouteType.GUARDRAIL.value

        return {
            "route": route,
            "overall_confidence": 0.5,
            "reasoning": "Fallback routing based on keyword analysis",
            "intent_analysis": {
                "primary_intent": "general_inquiry",
                "complexity_level": "MEDIUM",
                "requires_comparison": "비교" in lower_prompt,
                "requires_realtime_data": any(
                    k in lower_prompt for k in ["실시간", "현재"]
                ),
            },
            "needs_compliance_check": False,
        }

    def _format_profile_for_prompt(self, profile: Dict) -> str:
        """프로필 정보를 프롬프트용으로 포맷팅"""
        if not profile:
            return "No profile available"

        return f"""
- User ID: {profile.get("external_user_key", "Unknown")}
- Name: {profile.get("name_display", "Unknown")}
- Age Range: {profile.get("age_range", "Unknown")}
- Risk Tolerance: {profile.get("risk_tolerance_level", "Unknown")} (1-5 scale)
- Investment Experience: {profile.get("invest_experience_yr", 0)} years
- Income Bracket: {profile.get("income_bracket", "Unknown")}
- Investment Goal: {profile.get("goal_type", "Unknown")}
- Investment Style: {profile.get("preferred_style", "Unknown")}
- Investable Amount: {profile.get("total_investable_amt", 0):,} KRW
- Financial Knowledge: {profile.get("financial_knowledge_level", "Unknown")}
- Last Activity: {profile.get("updated_at", "Unknown")}
"""

    def _format_routing_result(self, result: Dict) -> str:
        """라우팅 결과를 읽기 쉬운 형태로 포맷팅"""
        return f"""
라우팅 결과:
- 선택된 경로: {result["route"]}
- 신뢰도: {result["context"]["overall_confidence"]:.1%}
- 주요 의도: {result.get("intent_analysis", {}).get("primary_intent", "Unknown")}
- 데이터 소스: {", ".join(result.get("data_sources", [])[:3])}
- 이유: {result.get("routing_reasoning", "N/A")}
"""

    async def _save_interaction_log(self, user_id: str, result: Dict):
        """상호작용 로그를 Supabase에 저장"""
        try:
            log_data = {
                "user_id": user_id,
                "route": result["route"],
                "intent": json.dumps(
                    result.get("intent_analysis", {}), ensure_ascii=False
                ),
                "confidence": result.get("context", {}).get(
                    "overall_confidence", Config.DEFAULT_CONFIDENCE
                ),
                "data_sources": json.dumps(
                    result.get("data_sources", []), ensure_ascii=False
                ),
                "compliance_check": json.dumps(
                    result.get("compliance_status", {}), ensure_ascii=False
                ),
                "created_at": datetime.now().isoformat(),
            }

            self.supabase.table("interaction_logs").insert(log_data).execute()

        except Exception as e:
            self.logger.error(f"Failed to save interaction log: {e}")


# ===== 테스트 헬퍼 =====
async def test_condition_agent():
    """독립 실행 테스트"""
    agent = ConditionAgentNode()

    test_cases = [
        "삼성전자와 SK하이닉스 중 어떤 주식이 더 나을까요?",
        "ETF가 뭔지 설명해 주세요",
        "내 포트폴리오 리밸런싱이 필요할까요?",
        "현재 코스피 지수와 환율 알려주세요",
        "은퇴자금 준비는 어떻게 해야 하나요?",
    ]

    for query in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")

        try:
            result = await agent.process_request(query, "test_user")
            print(f"Route: {result['route']}")
            print(f"Confidence: {result['context']['overall_confidence']:.1%}")
            print(f"Reasoning: {result['routing_reasoning'][:100]}...")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # 독립 실행 테스트
    import asyncio
    asyncio.run(test_condition_agent())