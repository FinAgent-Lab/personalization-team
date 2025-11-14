from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.graph.nodes.base import Node
from src.models.do import RawResponse
from src.models.user_profile import ExtractedInfo
from src.utils.const import PROMPT_USER_PROFILE_SURVEY, PROMPT_USER_PROFILE_CONTINUE
from src.services.conversation_state_manager import conversation_manager


class UserProfileChatNode(Node):
    """
    유저 프로필 수집을 위한 대화형 서베이 노드

    유저와 자연스러운 대화를 통해 투자 프로필 정보를 수집합니다.
    멀티턴 대화를 지원하며 상태를 추적합니다.
    """

    def __init__(self):
        super().__init__()
        self.system_prompt = PROMPT_USER_PROFILE_SURVEY
        self.agent = None
        self.tools = []  # 현재는 도구 없이 순수 대화만 진행

    def _run(self, state: dict) -> dict:
        """
        그래프 내에서 실행 (Supervisor → UserProfileChatNode → Supervisor)

        Args:
            state: LangGraph 상태 (llm, messages 포함)

        Returns:
            Command: supervisor로 돌아가는 명령
        """
        # 1. LLM 가져오기 (state에서 주입받은 LLM 사용)
        if self.agent is None:
            assert state["llm"] is not None, "The State model should include llm"
            llm = state["llm"]
            self.agent = create_react_agent(
                llm,
                self.tools,
                prompt=self.system_prompt,
            )

        # 2. 에이전트 실행
        result = self.agent.invoke(state)

        # 3. 결과 로깅
        self.logger.info(f"   result: \n{result['messages'][-1].content}")

        # 4. Supervisor로 결과 반환
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=result["messages"][-1].content,
                        name=self.__class__.__name__.lower().replace("node", ""),
                    )
                ]
            },
            goto="supervisor",
        )

    def _invoke(self, query: str, user_id: str = "default_user") -> RawResponse:
        """
        API 엔드포인트에서 직접 호출 (멀티턴 대화 지원)

        Args:
            query: 유저 입력 메시지
            user_id: 사용자 ID (대화 상태 추적용)

        Returns:
            RawResponse: 응답 메시지
        """
        # 1. 대화 상태 가져오기
        conv_state = conversation_manager.get_or_create(user_id)

        # 2. 대화 히스토리에 유저 메시지 추가
        conv_state.conversation_history.append({
            "role": "user",
            "content": query
        })

        # 3. LLM 생성 (Structured Output 사용)
        llm = ChatOpenAI(
            model=self.DEFAULT_LLM_MODEL,
            temperature=0.7,  # 대화체이므로 약간 높게
        )
        structured_llm = llm.with_structured_output(ExtractedInfo)

        # 4. 메시지 구성
        messages = self._build_messages(conv_state, query)

        # 5. Langfuse 콜백 설정
        config = self._get_callback_config()

        # 6. LLM 호출 (Structured Output)
        extracted_info: ExtractedInfo = structured_llm.invoke(messages, config=config)
        answer = extracted_info.response

        # 7. 추출된 필드를 상태에 병합
        extracted_dict = {
            k: v for k, v in extracted_info.extracted_fields.model_dump().items()
            if v is not None
        }
        if extracted_dict:
            conv_state.collected_fields.update(extracted_dict)
            self.logger.info(f"Extracted fields: {extracted_dict}")
            self.logger.info(f"Total collected: {conv_state.collected_fields}")

        # 8. 대화 히스토리에 AI 응답 추가
        conv_state.conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        # 9. 상태 업데이트
        conversation_manager.update(user_id, conv_state)

        # 10. 로깅
        self.logger.info(f"User {user_id}: {query}")
        self.logger.info(f"Assistant: {answer}")
        self.logger.info(f"Conversation turns: {len(conv_state.conversation_history)}")

        # 11. 완료 여부 확인
        missing_fields = conv_state.get_missing_fields()
        if not missing_fields:
            self.logger.info("✅ All profile information collected!")
            conv_state.is_completed = True

        # 10. 응답 반환
        return RawResponse(answer=answer)

    def _build_messages(self, conv_state, current_query: str):
        """
        LLM에 전달할 메시지 구성

        Args:
            conv_state: 현재 대화 상태
            current_query: 현재 유저 쿼리

        Returns:
            list: LangChain 메시지 리스트
        """
        messages = []

        # 첫 대화인지 확인
        is_first_turn = len(conv_state.conversation_history) <= 1

        if is_first_turn:
            # 첫 번째 턴: 초기 프롬프트 사용
            messages.append(SystemMessage(content=PROMPT_USER_PROFILE_SURVEY))
            messages.append(HumanMessage(content=current_query))
        else:
            # 이후 턴: 상태 기반 프롬프트 사용
            missing_fields = conv_state.get_missing_fields()
            collected_summary = self._format_collected_fields(conv_state.collected_fields)
            history_summary = self._format_conversation_history(conv_state.conversation_history[:-1])

            continue_prompt = PROMPT_USER_PROFILE_CONTINUE.replace(
                "{{collected_fields}}", collected_summary
            ).replace(
                "{{missing_fields}}", ", ".join(missing_fields) if missing_fields else "All information collected!"
            ).replace(
                "{{conversation_history}}", history_summary
            )

            messages.append(SystemMessage(content=continue_prompt))

            # 최근 대화 히스토리 추가 (마지막 4턴)
            recent_history = conv_state.conversation_history[-8:]  # 4턴 = 8개 메시지
            for msg in recent_history[:-1]:  # 현재 메시지 제외
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

            messages.append(HumanMessage(content=current_query))

        return messages

    def _format_collected_fields(self, collected: dict) -> str:
        """수집된 필드를 보기 좋게 포맷"""
        if not collected:
            return "None yet"

        lines = []
        for key, value in collected.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _format_conversation_history(self, history: list) -> str:
        """대화 히스토리를 보기 좋게 포맷"""
        if not history:
            return "No previous conversation"

        lines = []
        for msg in history[-6:]:  # 최근 3턴만
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
