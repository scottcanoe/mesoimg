import select
import sys
import termios
import tty

stdin = sys.stdin

# Characters
BACKSPACE = chr(8)
TAB = '\t'
NEWLINE = '\n'
ESCAPE = chr(27)
SPACE = ' '
UP_ARROW = '^[[A'


def poll_stdin(timeout: float = 0.0) -> bool:
    """
    Returns `True` if stdin has at least one line ready to read.
    """
    return select.select([stdin], [], [], timeout)[0] == [stdin]


orig_settings = termios.tcgetattr(stdin)
tty.setcbreak(stdin)

try:
    c = 0
    timeout = 0.001
    print('here', flush=True)
    while True:

        if not poll_stdin(timeout):
            continue
        print('here2')
        chars = []
        while poll_stdin(timeout):
            chars.append(stdin.read(1))

        print(f'chars: {chars}')

        if len(chars) == 1 and chars[0] == ESCAPE:
            break

except:
    termios.tcsetattr(stdin, termios.TCSADRAIN, orig_settings)
    raise

termios.tcsetattr(stdin, termios.TCSADRAIN, orig_settings)


