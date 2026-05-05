import re
import subprocess
from pathlib import Path

import requests
from langchain_core.tools import tool
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from ai_agent.config.settings import get_settings

# 用正则表达式粗暴剥离 HTML 标签、提取纯文本的工具函数，专门用来把网页源码转成干净的文字
# 真正生产环境建议用 BeautifulSoup 这类专业库
def _strip_html(html: str) -> str:
    text = re.sub(r"<script.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def get_manus_tools():
    settings = get_settings()
    file_dir = settings.workspace_dir / "files"
    download_dir = settings.workspace_dir / "downloads"
    pdf_dir = settings.workspace_dir / "pdf"
    file_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # @tool 的作用是把普通 Python 函数注册成 LangChain agent 可识别的工具
    # docstring 会变成工具描述，供模型理解何时该用它
    @tool
    def read_workspace_file(file_name: str) -> str:
        """Read a file from the agent workspace."""
        path = file_dir / Path(file_name).name
        try:
            return path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error reading file: {exc}"

    @tool
    def write_workspace_file(file_name: str, content: str) -> str:
        """Write content to a file in the agent workspace."""
        path = file_dir / Path(file_name).name
        try:
            path.write_text(content, encoding="utf-8")
            return f"File saved to {path}"
        except Exception as exc:
            return f"Error writing file: {exc}"

    @tool
    def search_web(query: str) -> str:
        """Search the web with SearchAPI."""
        if not settings.search_api_key:
            return "SEARCH_API_KEY is not configured."
        try:
            response = requests.get(
                "https://www.searchapi.io/api/v1/search",
                params={"q": query, "api_key": settings.search_api_key, "engine": "baidu"},
                timeout=20,
            )
            response.raise_for_status()
            organic_results = response.json().get("organic_results", [])[:5]
            return "\n".join(str(item) for item in organic_results) or "No search results."
        except Exception as exc:
            return f"Error searching web: {exc}"

    @tool
    def scrape_web_page(url: str) -> str:
        """Fetch a web page and return its visible text."""
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            return _strip_html(response.text)[:12000]
        except Exception as exc:
            return f"Error scraping web page: {exc}"

    @tool
    def download_resource(url: str, file_name: str) -> str:
        """Download a resource into the workspace download directory."""
        path = download_dir / Path(file_name).name
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
            return f"Downloaded resource to {path}"
        except Exception as exc:
            return f"Error downloading resource: {exc}"

    @tool
    def execute_terminal_command(command: str) -> str:
        """Run a terminal command inside the agent workspace directory."""
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(settings.workspace_dir),
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = (completed.stdout + completed.stderr).strip() or "(no output)"
            return f"exit_code={completed.returncode}\n{output[:12000]}"
        except Exception as exc:
            return f"Error executing command: {exc}"

    @tool
    def generate_pdf(file_name: str, content: str) -> str:
        """Generate a PDF file in the workspace PDF directory."""
        path = pdf_dir / Path(file_name).name
        try:
            pdf = canvas.Canvas(str(path), pagesize=A4)
            width, height = A4
            text = pdf.beginText(40, height - 50)
            text.setFont("Helvetica", 11)
            for line in content.splitlines() or [""]:
                text.textLine(line[:100])
            pdf.drawText(text)
            pdf.save()
            return f"PDF saved to {path}"
        except Exception as exc:
            return f"Error generating PDF: {exc}"

    @tool
    def terminate() -> str:
        """Use this tool when the task is complete and you should stop using tools."""
        return "任务完成。请整理最终答复并停止继续调用工具。"

    return [
        read_workspace_file,
        write_workspace_file,
        search_web,
        scrape_web_page,
        download_resource,
        execute_terminal_command,
        generate_pdf,
        terminate,
    ]
