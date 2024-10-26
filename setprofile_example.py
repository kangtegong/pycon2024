import sys

def profile_func(frame, event, arg):
    if event == "call":
        print(f"[Call] {frame.f_code.co_name}")
    elif event == "return":
        print(f"[Return] {frame.f_code.co_name}")
    return profile_func

def bar():
    pass

def foo():
    bar()

def main():
    foo()
    bar()

if __name__ == "__main__":
    sys.setprofile(profile_func)
    main()
    sys.setprofile(None)