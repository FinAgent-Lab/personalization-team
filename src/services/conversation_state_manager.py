"""대화 상태 관리 서비스"""

from typing import Dict
from src.models.user_profile import ConversationState


class ConversationStateManager:
    """
    In-memory 대화 상태 저장소

    유저별 대화 상태를 메모리에 저장하고 관리합니다.
    프로덕션 환경에서는 Redis나 DB로 교체 가능합니다.
    """

    def __init__(self):
        self._states: Dict[str, ConversationState] = {}

    def get_or_create(self, user_id: str) -> ConversationState:
        """
        유저 ID로 대화 상태를 가져오거나 새로 생성

        Args:
            user_id: 사용자 ID

        Returns:
            ConversationState: 대화 상태 객체
        """
        if user_id not in self._states:
            self._states[user_id] = ConversationState(user_id=user_id)
        return self._states[user_id]

    def update(self, user_id: str, state: ConversationState) -> None:
        """
        대화 상태 업데이트

        Args:
            user_id: 사용자 ID
            state: 업데이트할 상태
        """
        self._states[user_id] = state

    def delete(self, user_id: str) -> None:
        """
        대화 상태 삭제 (대화 완료 시)

        Args:
            user_id: 사용자 ID
        """
        if user_id in self._states:
            del self._states[user_id]

    def clear_all(self) -> None:
        """모든 대화 상태 삭제"""
        self._states.clear()


# Singleton 인스턴스
conversation_manager = ConversationStateManager()
