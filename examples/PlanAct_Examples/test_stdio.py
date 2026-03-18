import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys

async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["examples/PlanAct_Examples/sample_mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tool = "mul"
            args = {"a": 2, "b": 5}
            result = await session.call_tool(tool, args)

            structured = getattr(result, "structuredContent", None)

            if structured is not None:
                print("structuredContent:", structured)
                if isinstance(structured, dict):
                    print("product =", structured.get("result"))
                    print("type:", type(structured.get("result")))
            else:
                print("No structuredContent was returned.")
                for block in result.content:
                    if getattr(block, "type", None) == "text":
                        print(block.text)

if __name__ == "__main__":
    asyncio.run(main())