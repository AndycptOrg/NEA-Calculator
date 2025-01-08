# About
## Introduction
This is my 22-24 A-Level AQA NEA

The intended outcome of this NEA is to create a complete basic calculator backend API

### Acknowledgement
Before we continue, I would like to thank my Computer Science teacher, Dr Wild, for overseeing my project.
The reason why this NEA is the way that is is now is in large part thanks to him.
In addition to overseeing my project, he has also been a great supporter and believer in my abilities and pushed me to pursue a more challenging topic for my NEA.
There are many more things that he has done to propel my journey, but in short

Thank You.

### Getting it Running
As an extended goal, a command line and (mininal) graphical interface has been provided when you run the main functions in `code/user_interface.py` and `code/tktinkergraphics.py` respectively

### Writeup
As you can see, the writeup has been included in `writeup.docx`, which you can read through for a more detailed description of the program/project

### Diagrams
In the Syntax Diagrams folder, there is the source code used to generate the diagrams in the writeup,
the sites used to generate all the diagrams have been included in the bibliography of the writeup if you are interested

### `log.txt`
This file is generated each time the tests for the lexer or parser is run (we'll get to that later) and shows if the code passes the specifed tests

## Background & Warnings
The program is **not** intended to be used as an actuall calculator backend and more so serves as a proof of concept (although if you really wanted to, who am I to stop you)

The program utilizes many concepts that are beyond the A-Level computer science syllabus and is probably not suitable for most,
but for those that have gone beyond the syllabus, this program and it's concepts may be of interest to you

This was written in the midst of my functional programming phase, so you may notice heavy functional style used throughout the program

There is a lot of OOP code which when examined carefully, is actually immutable

### Monads
The concpet of monads is not nessecary to the understanding of how the program operates, nor is it critical to the program flow

### Reading Input (parsing)
The way I have decided to read and convert text input from the interface was using a concept called parsing

To go into a bit of the deails, 
1. the program converts the text input into "nuggets" of independent information (tokens) via a tokenizer
2. the program then constructs from the result of that operation a tree via a parser, which encapsulates the operations intended in the text
3. the parsing stratagy used is commonly described as `Top Down`

#### Top Down Parsing
This stratagy is where the sequence of tokens is matched from left to right from the overarching rule and seeing if the sequence matches the pattern laid out by the syntax rules.

When parsing a top down language, there is usually one rule that describes all valid sequence of tokens. So:
1. take the left most token in the sequence and check against the overarching rule and any composed rules
2. examine any further composed rules until a terminal rule is reached (i.e. a rule specifying a token)
3. repeat with the next token in the sequence progressing in the rule

refer to Wiki
[or other sources](https://www.geeksforgeeks.org/working-of-top-down-parser/)
for better explanations that do a far better justice to this simple stratagy

### Abstract Syntax Tree
A tree struture that is used to store syntactic structure of a language

In my case, I have chosen to embed the structre of the tree into the parsed result, i.e. as the syntactic structure is determined, the nodes of the tree are formed in the same time and are inseperable

### OOP Design Pattern (Visitor Pattern)
This is designing nested objects to standardize the access of all internal objects

This was used to factor out recursive methods 

I've done it by implementing a unified (visitor and Visited) interface (though you don't really have to do it in python since python checks if the method exists at runtime)

## Overview
Once the program recieves the text input(by being called),
1. it will be tokenized and parsed into an abstract syntax tree
2. the tree will be evaluated
3. any internal state that should be updated will be updated
4. the result will be returned to the interface

## Looking Back
- If I had anything to say about the choice of the language used to write the program, python would not be at the top of the list
- The implementation of the syntax tree could have been done a bit differently, decoupling the actual elements from the tree itself
- learn about @dataclass
- find someone you trust and ask them for project ideas, spent a bit too long meandering around topics

Other than that, I absolutely loved this project, and if I could have time to redo it, I would absolutely do it again 
