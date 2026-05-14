import asyncio
from sandbox_fusion import (
    RunCodeRequest,
    RunStatus,
    run_code_async,
    set_sandbox_endpoint,
)


async def single_sandbox(
    code, stdin: str = None, language="python", compile_timeout=1.0, run_timeout=3.0, semaphore=None
):
    request = RunCodeRequest(
        code=code,
        stdin=stdin,
        language=language,
        compile_timeout=compile_timeout,
        run_timeout=run_timeout,
    )
    async with semaphore:
        response = await run_code_async(request, client_timeout=30.0, max_attempts=2)
        response = response.dict()
    await asyncio.sleep(2)
    return response


async def parallel_sandbox(tasks, stdin_list=None, num_processes=200):
    semaphore = asyncio.Semaphore(num_processes)
    set_sandbox_endpoint("https://seed-sandbox.byteintl.net/faas/sandbox/")
    if stdin_list is None:
        tasks_async = [single_sandbox(code, semaphore=semaphore) for code in tasks]
    else:
        assert len(tasks) == len(stdin_list), f"len(tasks) ({len(tasks)}) != len(stdin_list) ({len(stdin_list)})"
        tasks_async = [single_sandbox(code, stdin, semaphore=semaphore) for code, stdin in zip(tasks, stdin_list)]
    results = await asyncio.gather(*tasks_async, return_exceptions=True)
    
    ok_flags = []
    stdouts = []
    stderrs = []
    
    for r in results:
        if isinstance(r, Exception):
            # Handle exception: single sandbox failure should not affect others
            ok_flags.append(False)
            stdouts.append("")
            stderrs.append(f"Sandbox error: {str(r)}")
        else:
            ok_flags.append(r["status"] == RunStatus.Success)
            stdouts.append(r["run_result"]["stdout"])
            stderrs.append(r["run_result"]["stderr"])
    
    return ok_flags, stdouts, stderrs


if __name__ == "__main__":
    code_list = ["""
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print([fib(i) for i in range(10)])
""", """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b, v
    return a

print([fib(i) for i in range(10)])
""", """
name = input("Your name:")
print(f"Hi, {name}!")
"""]
    stdin_list = ["", "", "Alice"]

    for code, stdin in zip(code_list, stdin_list):
        print(f"code: {code}")
        sandbox_success, sandbox_stdout, sandbox_stderr = asyncio.run(parallel_sandbox(tasks=[code], stdin_list=[stdin]))

        print(f"sandbox_success: {sandbox_success}")
        print(f"sandbox_stdout: {sandbox_stdout}")
        print(f"sandbox_stderr: {sandbox_stderr} {sandbox_stderr[0].splitlines()[-1:]}")
