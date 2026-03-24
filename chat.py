import logging
logging.disable(logging.CRITICAL)

from app.ask import stream_ask

# ANSI colors
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
GRAY   = "\033[90m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

BANNER = r"""
  ____                          ____      _    ____
 |  _ \ __ _ _ __   ___ _ __  |  _ \    / \  / ___|
 | |_) / _` | '_ \ / _ \ '__| | |_) |  / _ \| |  _
 |  __/ (_| | |_) |  __/ |    |  _ <  / ___ \ |_| |
 |_|   \__,_| .__/ \___|_|    |_| \_\/_/   \_\____|
             |_|
         Research Assistant — ask anything about your papers
"""

print(f"{CYAN}{BOLD}{BANNER}{RESET}")
print(f"{GRAY}  Type your question and press Enter. Ctrl+C to exit.\n{RESET}")
print(f"{GRAY}{'─' * 60}{RESET}")

while True:
    try:
        print()
        question = input(f"  {GREEN}{BOLD}You ›{RESET} {GREEN}").strip()
        print(RESET, end="", flush=True)
        if not question:
            continue
        print()
        print(f"  {CYAN}{BOLD}Assistant ›{RESET} ", end="", flush=True)
        try:
            stream_ask(question, color=CYAN, reset=RESET)
        except Exception as e:
            print(f"{YELLOW}[Error] {e}{RESET}")
        print()
        print(f"{GRAY}{'─' * 60}{RESET}")
    except KeyboardInterrupt:
        print(f"\n\n  {GRAY}Goodbye.{RESET}\n")
        break
