import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    sse_url = "http://localhost:8000/sse"  # legacy SSE endpoint

    async with sse_client(sse_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tool = "mul" # give tool function name
            args = {"a": 2, "b": 5} # format inputs as key-word-only dict
            result = await session.call_tool(tool, args)

            # This is the machine-readable field you want
            structured = getattr(result, "structuredContent", None)

            if structured is not None:
                print("structuredContent:", structured)

                # By default, structured content is formatted {"result": Any}
                # But if your output is a structured type, something serializable
                # is best, because it returns a dict, for instance if you return
                # a dataclass, it will expect a data-class object
                if isinstance(structured, dict):
                    print("product =", structured.get("result"))
                    print("type:", type(structured.get("result")))
            else:
                print("No structuredContent was returned.")
                print("Fallback text blocks:")
                for block in result.content:
                    if getattr(block, "type", None) == "text":
                        print(block.text)

if __name__ == "__main__":
    asyncio.run(main())