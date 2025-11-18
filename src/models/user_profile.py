"""
유저 프로필 데이터 모델

수집 필드 (총 12개):
- 기본 정보: name_display, age_range, income_bracket
- 투자 경험: invest_experience_yr, financial_knowledge_level
- 투자 성향: risk_tolerance_level (1~5등급), preferred_style
- 투자 목표: goal_type, goal_description
- 자산 정보: total_investable_amt, current_holdings_note, preferred_asset_types

위험등급 체계 (risk_tolerance_level):
- 1등급 (초저위험형): 국공채형, MMF 등 선호
- 2등급 (저위험형): 채권형, 원금보존 추구형 ELF/DLF 선호
- 3등급 (중위험형): 채권혼합형, 원금부분보존 추구형 ELF/DLF 선호
- 4등급 (고위험형): 주식혼합형, 인덱스펀드, 원금비보장형 ELF/DLF 선호
- 5등급 (초고위험형): 주식형, 파생형 등 고위험 상품 선호
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """
    유저 프로필 데이터 모델 (DB 저장용)

    총 12개 필드 수집 (created_at, updated_at 제외)
    """

    user_id: str = Field(description="사용자 고유 ID")
    external_user_key: Optional[str] = Field(
        default=None, description="외부 시스템 연동 키"
    )
    name_display: Optional[str] = Field(default=None, description="표시 이름")
    age_range: Optional[str] = Field(
        default=None, description="연령대 (예: 20-29, 30-39)"
    )
    income_bracket: Optional[str] = Field(
        default=None, description="소득 구간 (예: 50M-100M)"
    )
    invest_experience_yr: Optional[int] = Field(
        default=None, description="투자 경험 연수"
    )
    risk_tolerance_level: Optional[str] = Field(
        default=None,
        description="위험등급 (1등급: 초저위험형, 2등급: 저위험형, 3등급: 중위험형, 4등급: 고위험형, 5등급: 초고위험형)",
    )
    # NOTE: LLM이 "저위험형" 등으로 반환 가능. UserProfileChatNode에서 자동 정규화됨
    goal_type: Optional[str] = Field(
        default=None, description="투자 목표 타입 (예: retirement, wealth_building)"
    )
    goal_description: Optional[str] = Field(
        default=None, description="투자 목표 상세 설명"
    )
    preferred_style: Optional[str] = Field(
        default=None, description="선호하는 투자 스타일"
    )
    total_investable_amt: Optional[float] = Field(
        default=None, description="총 투자 가능 금액"
    )
    current_holdings_note: Optional[str] = Field(
        default=None, description="현재 보유 자산 메모"
    )
    preferred_asset_types: Optional[list[str]] = Field(
        default=None, description="선호하는 자산 유형들"
    )
    financial_knowledge_level: Optional[str] = Field(
        default=None, description="금융 지식 수준"
    )
    created_at: Optional[datetime] = Field(
        default_factory=datetime.now, description="생성 일시"
    )
    updated_at: Optional[datetime] = Field(
        default_factory=datetime.now, description="수정 일시"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "external_user_key": "ext_key_abc",
                "name_display": "김투자",
                "age_range": "30-39",
                "income_bracket": "50M-100M",
                "invest_experience_yr": 3,
                "risk_tolerance_level": "3등급",
                "goal_type": "retirement",
                "goal_description": "55세 은퇴를 목표로 안정적인 노후 자금 마련",
                "preferred_style": "balanced",
                "total_investable_amt": 50000000.0,
                "current_holdings_note": "예금 3천만원, 주식 2천만원",
                "preferred_asset_types": ["stocks", "bonds", "etf"],
                "financial_knowledge_level": "intermediate",
            }
        }


class ExtractedFields(BaseModel):
    """추출된 필드들"""

    name_display: Optional[str] = None
    age_range: Optional[str] = None
    income_bracket: Optional[str] = None
    invest_experience_yr: Optional[int] = None
    risk_tolerance_level: Optional[str] = None
    goal_type: Optional[str] = None
    goal_description: Optional[str] = None
    preferred_style: Optional[str] = None
    total_investable_amt: Optional[float] = None
    current_holdings_note: Optional[str] = None
    preferred_asset_types: Optional[list[str]] = None
    financial_knowledge_level: Optional[str] = None

    class Config:
        extra = "forbid"


class ExtractedInfo(BaseModel):
    """
    대화에서 추출된 정보
    LLM이 Structured Output으로 반환
    """

    response: str = Field(description="유저에게 보낼 응답 메시지")
    extracted_fields: ExtractedFields = Field(
        default_factory=ExtractedFields, description="이번 턴에서 추출된 필드들"
    )

    class Config:
        extra = "forbid"  # additionalProperties: false


class ConversationState(BaseModel):
    """대화 상태 추적용 모델"""

    user_id: str
    collected_fields: dict = Field(default_factory=dict, description="수집된 필드들")
    current_question: Optional[str] = Field(default=None, description="현재 질문")
    conversation_history: list[dict] = Field(
        default_factory=list, description="대화 히스토리"
    )
    is_completed: bool = Field(default=False, description="수집 완료 여부")

    def get_missing_fields(self) -> list[str]:
        """
        아직 수집하지 못한 필드 목록 반환

        NOTE: 이 메서드는 key 존재 여부만 확인합니다.
        빈 값 체크는 UserProfileChatNode._get_truly_missing_fields()를 사용하세요.
        """
        all_fields = [
            "name_display",
            "age_range",
            "income_bracket",
            "invest_experience_yr",
            "risk_tolerance_level",
            "goal_type",
            "goal_description",
            "preferred_style",
            "total_investable_amt",
            "current_holdings_note",
            "preferred_asset_types",
            "financial_knowledge_level",
        ]
        return [f for f in all_fields if f not in self.collected_fields]
