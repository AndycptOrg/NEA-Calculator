import string
import tkinter as tk
from tkinter import ttk
from controlunit import *
from mathobj import *
import time

root = tk.Tk()
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

main_frame = ttk.Frame(root)
main_frame.grid(row=0, column=0, sticky="nesw")
for row in range(3):
    main_frame.rowconfigure(row, weight=1)
main_frame.columnconfigure(0, weight=1)

user_input_display = tk.StringVar()
label = ttk.Label(main_frame, textvariable=user_input_display)
label.grid(column=0, row=0)
# ttk.Button(main_frame, text="Quit", command=root.destroy).grid(column=1, row=0)


"""
https://www.astro.princeton.edu/~rhl/Tcl-Tk_docs/tk8.0a1/bind.n.html
pie()Ac
789^Del
456x/
123+-
0.,AnsExc
https://tkdocs.com/tutorial/grid.html
https://tkdocs.com/shipman/key-names.html
https://tkdocs.com/shipman/root-resize.html
"""
ttk.Separator(main_frame, orient="horizontal").grid(row=1, column=0)
button_pad = ttk.Frame(main_frame)
button_pad.grid(row=2, column=0)
for row in range(5):
    button_pad.rowconfigure(row, weight=1)
for column in range(5):
    button_pad.columnconfigure(column, weight=1)
# button_pad.rowconfigure(0, weight=1)
# button_pad.columnconfigure(0, weight=1)

# prints number to console
command = (lambda x: (lambda: print(str(x))))


altered_by_program = False
# adds to end of label
def add_to_display(x):
    def _():
        global altered_by_program
        if not altered_by_program:
            user_input_display.set(user_input_display.get() + str(x))
        else:
            if str(x) in "+-*÷^|":
                user_input_display.set("Ans"+str(x))
            else:
                user_input_display.set(str(x))
            altered_by_program = False
    return _


for i in range(9):
    number = 9-i
    ttk.Button(button_pad, text=str(number), command=add_to_display(number)).grid(row=i // 3 + 1, column=2 - (i % 3))
# pi button
ttk.Button(button_pad, text="π", command=add_to_display("π")).grid(row=0, column=0)
# e button
ttk.Button(button_pad, text="e", command=add_to_display("e")).grid(row=0, column=1)
# open( bracket
ttk.Button(button_pad, text="(", command=add_to_display("(")).grid(row=0, column=2)
# close bracket
ttk.Button(button_pad, text=")", command=add_to_display(")")).grid(row=0, column=3)
# clear button
ttk.Button(button_pad, text="AC", command=lambda: user_input_display.set("")).grid(row=0, column=4)
# delete button
ttk.Button(button_pad, text="DEL", command=lambda: user_input_display.set(user_input_display.get()[:-1])).grid(row=1, column=4)
# 0 button
ttk.Button(button_pad, text="0", command=add_to_display(0)).grid(row=4, column=0)
# . button
ttk.Button(button_pad, text=".", command=add_to_display(".")).grid(row=4, column=1)
# , button
ttk.Button(button_pad, text=",", command=add_to_display(",")).grid(row=4, column=2)
# adding operation buttons
for i, operation in enumerate("x÷+-"):
    ttk.Button(button_pad, text=operation, command=add_to_display(operation)).grid(row=i // 2 + 2, column=i % 2 + 3)
# power button
ttk.Button(button_pad, text="^", command=add_to_display("^")).grid(row=1, column=3)
# ans button
ttk.Button(button_pad, text="Ans", command=add_to_display("Ans")).grid(row=4, column=3)


class TKInterGUI(UserInterfaceInterface):
    def __init__(self):
        super(TKInterGUI, self).__init__()

    def alert_improper_input(self, msg):
        global altered_by_program
        user_input_display.set(msg)
        print("alert", msg)
        altered_by_program = True


interface = TKInterGUI()

def value_matcher(value: Value):
    match value:
        case Undefined():
            return "Undefined"
        case Number(_val=num):
            return num
        case _:
            return "Unknown type"


def execute_subroutine(event=None):
    global altered_by_program
    print(user_inp := user_input_display.get())
    interface.set_instruction(user_inp.replace("÷", "/").replace("π", "pi")+";")
    calculated = interface.get_calculated_results()
    print("calculated", calculated)
    if user_input_display.get() == user_inp and len(calculated) >= 1:
        user_input_display.set(value_matcher(calculated[0]))
        altered_by_program = True


# execute button
ttk.Button(button_pad, text="EXE", command=execute_subroutine).grid(row=4, column=4)


def bind_keypress(bind_to=None, **mappings: dict):
    if bind_to is None:
        bind_to = root
    for key, command in mappings.items():
        bind_to.bind(key, command)


# numbers keypress binding
bind_keypress(**{str(i): (lambda event: add_to_display(event.keysym)()) for i in range(10)},
              **{
                  # enter -> EXE
                  '<Return>': execute_subroutine,
                  # ctr Backspace -> AC
                  '<Control-BackSpace>': lambda _: user_input_display.set(""),
                  # Backspace -> DEL
                  '<BackSpace>': lambda _: user_input_display.set(user_input_display.get()[:-1])
              },
              **{  # buggy, only works when race condition triggered
                  '<p>i': lambda _: add_to_display("π")()
                },
              # key if len(key := event.keysym) == 1 else {"parenleft": "(","parenright": ")", "comma":",","period":".", "slash":"÷","minus":"-","plus":"+","asterisk":"*", "asciicircum":"^", "bar":"|"}[event.keysym]
              **{letter: lambda event: add_to_display({"/": "÷"}.get(event.char, event.char))() for letter in string.ascii_letters + "()+-*/,.^|=~"},
              bind_to=main_frame
              )


# bind esc and del to closing window
for key in ('<Escape>', '<Delete>'):
    root.bind(key, lambda _: root.destroy())

main_frame.focus_set()
root.mainloop()
