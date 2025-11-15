from dotenv import load_dotenv

from dependency_injector.wiring import Provide, inject
import uvicorn

# from apscheduler.schedulers.background import BackgroundScheduler

from api.server import APIBuilder

# from src.graph.nodes.us_financial_fmg import StockInfoNode
# from src.graph.nodes import (
#     NaverNewsSearcherNode,
#     USFinancialAnalyzerNode,
# )
from src.graph.nodes.rss_feeder import ChosunRSSFeederNode
from src.graph.nodes.user_profile_chat import UserProfileChatNode
from src.graph.nodes.condition_agent import ConditionAgentNode
from src.utils.logger import setup_logger
from src.graph.builder import SupervisorGraphBuilder

# from src.tasks.weekly_recap_scraper import scrape_jp_weekly_recap
from startup import Container
from rich.console import Console

# import os


console = Console()
load_dotenv(override=True)
logger = setup_logger("personalization_agent")
logo = r"""
[cyan]
==============================================================================================

███████ ██ ███    ██  █████   ██████  ███████ ███    ██ ████████       ██       █████  ██████  
██      ██ ████   ██ ██   ██ ██       ██      ████   ██    ██          ██      ██   ██ ██   ██ 
█████   ██ ██ ██  ██ ███████ ██   ███ █████   ██ ██  ██    ██    █████ ██      ███████ ██████  
██      ██ ██  ██ ██ ██   ██ ██    ██ ██      ██  ██ ██    ██          ██      ██   ██ ██   ██ 
██      ██ ██   ████ ██   ██  ██████  ███████ ██   ████    ██          ███████ ██   ██ ██████  
                                                                                               
----------------------------------------------------------------------------------------------
                                                  _ _          _   _             
                                                 | (_)        | | (_)            
            _ __   ___ _ __ ___  ___  _ __   __ _| |_ ______ _| |_ _  ___  _ ___  
           | '_ \ / _ \ '__/ __|/ _ \| '_ \ / _` | | |_  / _` | __| |/ _ \| '_  | 
           | |_) |  __/ |  \__ \ (_) | | | | (_| | | |/ / (_| | |_| | (_) | | | |
           | .__/ \___|_|  |___/\___/|_| |_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
           | |                                                                   
           |_|                                                                       
----------------------------------------------------------------------------------------------
# MEMBER(가나다 순)
- 김기록 
- 김소영
- 박수형
- 배효영
- 엄창용        https://github.com/e7217
- 장현상
- 최재혁
----------------------------------------------------------------------------------------------
                                                    Since 2025.09.07, Let's study together!
==============================================================================================
"""


@inject
def main(
    graph_builder: SupervisorGraphBuilder = Provide[Container.supervisor_graph],
):
    console.print(logo)
    logger.info("Starting Market Analysis Agent service...")

    ## 그래프 빌더
    """
    에이전트 노드를 이곳에 추가해주세요.
    노드 추가 시, 다음의 기능을 동적으로 적용하게됩니다.
    - supervisor 노드에 멤버로 등록
    - 노드 이름을 기반으로 API 엔드포인트 생성(예: SampleNode -> /api/sample)
    
    Example:
    graph_builder.add_node(NewNode())
    """

    # graph_builder.add_node(NaverNewsSearcherNode())
    graph_builder.add_node(ChosunRSSFeederNode())
    graph_builder.add_node(UserProfileChatNode())
    # graph_builder.add_node(USFinancialAnalyzerNode())  # TODO: 종합 처리 기능 적용 시 주석 해제

    graph_builder.add_node(ConditionAgentNode())

    graph_builder.build()

    ## API 서버 빌더
    api_builder = APIBuilder()
    app = api_builder.create_app()
    for node in graph_builder.get_nodes():
        app.add_api_route(
            f"/api/{node.__class__.__name__.lower().replace('node', '')}",
            methods=["POST"],
            endpoint=node.invoke,
        )
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    container = Container()
    container.wire(
        modules=[
            __name__,
            "api.route",
        ]
    )
    main()
