import sys

# 첫 번째 도구: 간단한 디버거
def debugger(frame, event, arg):
    if event == 'line':
        print(f"[Debugger] {event}: {frame.f_code.co_name}")
    return debugger

# 두 번째 도구: 간단한 프로파일러
def profiler(frame, event, arg):
    if event == 'call':
        print(f"[Profiler] {event}: {frame.f_code.co_name}")
    return profiler

# 세 번째 도구: 테스트 커버리지 도구
def coverage(frame, event, arg):
    if event == 'return':
        print(f"[Coverage] {event}: {frame.f_code.co_name}")
    return coverage


def bar():
    print("hello bar")
    return

def foo():
    bar()
    print("hello foo")
    return


if __name__ == '__main__':
    sys.settrace(debugger)
    sys.settrace(profiler)
    sys.settrace(coverage)

    foo()

    sys.settrace(None)