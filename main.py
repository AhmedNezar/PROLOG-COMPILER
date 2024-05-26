import sys
import re
from enum import Enum
import tkinter as tk
from tkinter import ttk
import pandas as pd
import pandastable as pt
from nltk.tree import *
from graphviz import Digraph
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import random


class Token_type(Enum):
    Predicates = 1
    Clauses = 2
    Goal = 3
    Predicate_name = 4
    Data_type = 5
    Comma = 6
    LeftParenthesis = 7
    RightParenthesis = 8
    Colon = 9
    Semicolon = 10
    Period = 11
    Underscore = 12
    Integer = 13
    Char = 14
    String = 15
    Symbol = 16
    Real = 17
    Whitespace = 18
    Error = 19
    Plus = 20
    Minus = 21
    Multiply = 22
    Divide = 23
    Equal = 24
    Less_than = 25
    Greater_than = 26
    Less_than_equal = 27
    Greater_than_equal = 28
    Variable = 29
    If = 30
    Difference = 31
    Fact = 32
    Rule = 33
    Head = 34
    Annonymous_variable = 35
    Readln = 36
    Readint = 37
    Readchar = 38
    Write = 39
    nl = 40
    Exclamation = 41


class token:
    def __init__(self, lex, token_type):
        self.lex = lex
        self.token_type = token_type

    def to_dict(self):
        return {
            'Lex': self.lex,
            'token_type': self.token_type
        }


ReservedWords = {"predicates": Token_type.Predicates,
                 "clauses": Token_type.Clauses,
                 "goal": Token_type.Goal,
                 "readln": Token_type.Readln,
                 "readint": Token_type.Readint,
                 "readchar": Token_type.Readchar,
                 "write": Token_type.Write
                 }

DataTypes = {"string": Token_type.String,
             "integer": Token_type.Integer,
             "char": Token_type.Char,
             "real": Token_type.Real,
             "symbol": Token_type.Symbol
             }

Operators = {".": Token_type.Period,
             ";": Token_type.Semicolon,
             ",": Token_type.Comma,
             "=": Token_type.Equal,
             ":=": Token_type.If,
             "+": Token_type.Plus,
             "-": Token_type.Minus,
             "*": Token_type.Multiply,
             "/": Token_type.Divide,
             ">": Token_type.Greater_than,
             "<": Token_type.Less_than,
             ">=": Token_type.Greater_than_equal,
             "<=": Token_type.Less_than_equal,
             "<>": Token_type.Difference,
             "(": Token_type.LeftParenthesis,
             ")": Token_type.RightParenthesis,
             "!": Token_type.Exclamation
             }

Tokens = []  # to add tokens to list
errors = []
comment = False


def print_tokens():
    for token_obj in Tokens:
        print(token_obj.lex, token_obj.token_type)
    Token_Type = Tokens[0].to_dict()
    for x in range(0, len(Tokens)):
        TokenType = Tokens[x].to_dict()
        print(TokenType['token_type'], x+1)


def find_token(text):
    split_text = re.findall(
        r'[+-]?[0-9]+[.][0-9]+|\w+\s*\w*|[()]|"[^"]*"|\'[^\']*[\']|\/\*|\*\/|:-|:=|>=|<=|<>|[,;.\-+=\*/><!]|%.*', text)
    bracket_opened = False
    global comment
    print(split_text)
    for word in split_text:
        word = word.strip()
        if comment:  # if comment is true then skip all words until you find "*/"
            if word == "*/":
                comment = False
            else:
                continue
        elif word in ReservedWords:
            Tokens.append(token(word, ReservedWords[word]))
        elif word in DataTypes:
            # Tokens.append(token(word, DataTypes[word]))
            Tokens.append(token(word, Token_type.Data_type))
        elif word in Operators:
            if word == "(":
                bracket_opened = True
            elif word == ")":
                bracket_opened = False
            Tokens.append(token(word, Operators[word]))
        elif word == "_":
            Tokens.append(token(word, Token_type.Annonymous_variable))
        elif re.search("^[A-Z_][a-zA-z0-9_]*$", word):
            Tokens.append(token(word, Token_type.Variable))
        elif re.search("^[a-z][a-zA-z0-9_]*$", word):
            if word == "nl":
                Tokens.append(
                    token(word, Token_type.Symbol if bracket_opened else Token_type.nl))
            else:
                Tokens.append(
                    token(word, Token_type.Symbol if bracket_opened else Token_type.Predicate_name))
        elif re.search("^\".*\"$", word):
            # string between double quotes
            Tokens.append(token(word, Token_type.String))
        elif re.search("^'[a-zA-z0-9]?'$", word):
            # char between single quotes & only one char
            Tokens.append(token(word, Token_type.Char))
        elif re.search("^[+-]?[0-9]+$", word):
            Tokens.append(token(word, Token_type.Integer))
        elif re.search("^[+-]?[0-9]+[.][0-9]+$", word):
            Tokens.append(token(word, Token_type.Real))
        elif re.search("^:-$", word):
            Tokens.append(token(word, Token_type.Head))
        elif re.search("^(%|\/\*)", word):
            if (word == "/*"):
                comment = True
            else:
                break
        else:
            Tokens.append(token(word, Token_type.Error))
            errors.append("Lexical error  " + word)


pass


def scan_prolog(code):
    # split code into lines and remove any trailing or leading whitespace
    global comment
    lines = [line.strip() for line in code.split('\n')]
    # loop through each line and identify the section it belongs to
    for line in lines:
        # skip empty lines and comments
        if not line or line.startswith('%'):
            continue
        if comment and not "*/" in line:
            continue
        find_token(line)
    print_tokens()


def Parse():
    j = 0
    Children = []
    Predicates = Match(Token_type.Predicates, j)
    Children.append(Predicates["node"])
    predicates_dict = predicates(Predicates["index"])
    Children.append(predicates_dict["node"])
    Clauses = Match(Token_type.Clauses, predicates_dict["index"])
    Children.append(Clauses["node"])
    clauses_dict = clauses(Clauses["index"])
    Children.append(clauses_dict["node"])
    Goal = Match(Token_type.Goal, clauses_dict["index"])
    Children.append(Goal["node"])
    goal_dict = goal(Goal["index"])
    Children.append(goal_dict["node"])
    if (goal_dict["index"] < len(Tokens)):
        for x in range(goal_dict["index"], len(Tokens)):
            TokenType = Tokens[x].to_dict()
            st = str(TokenType['token_type'])
            errors.append(
                "Syntax error: Expected end of input but found " + st)
    Node = Tree('Program', Children)
    return Node


def predicates(j):
    predicates_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        if (Token_Type['token_type'] != Token_type.Clauses):
            predicate_dict = predicate(j)
            Children.append(predicate_dict["node"])
            predicates_dict = predicates(predicate_dict["index"])
            Children.append(predicates_dict['node'])
            Node = Tree("predicates", Children)
            predicates_output["node"] = Node
            predicates_output["index"] = predicates_dict["index"]
            return predicates_output
        else:
            Node = Tree('predicates', [])
            predicates_output['node'] = Node
            predicates_output['index'] = j
            return predicates_output
    else:

        # Node = Tree('predicates', [])
        # predicates_output['node'] = Node
        predicates_output['node'] = ['error']
        predicates_output['index'] = j
        return predicates_output


def predicate(j):
    predicate_output = dict()
    Children = []
    Token_Type = Tokens[j].to_dict()
    out1 = Match(Token_type.Predicate_name, j)
    print(Token_Type['token_type'], j)
    Children.append(out1["node"])
    predicate_output["index"] = out1["index"]
    if (out1["index"] < len(Tokens)):
        Token_Type = Tokens[out1["index"]].to_dict()
        if (Token_Type['token_type'] == Token_type.LeftParenthesis):
            out2 = Match(Token_type.LeftParenthesis, out1["index"])
            Children.append(out2["node"])
            data_types_dict = data_types(out2["index"])
            Children.append(data_types_dict["node"])
            out3 = Match(Token_type.RightParenthesis, data_types_dict["index"])
            Children.append(out3["node"])
            predicate_output["index"] = out3["index"]
            Node = Tree("predicate", Children)
            predicate_output["node"] = Node
            return predicate_output
        else:
            Node = Tree("predicate", Children)
            predicate_output["node"] = Node
            predicate_output['index'] = out1['index']
            return predicate_output
    else:
        Node = Tree("predicate", Children)
        predicate_output["node"] = Node
        predicate_output['index'] = out1['index']
        return predicate_output


def data_types(j):
    data_types_output = dict()
    Children = []
    data_type_dict = data_type(j)
    Children.append(data_type_dict["node"])
    data_types_tail_dict = data_types_tail(data_type_dict["index"])
    Children.append(data_types_tail_dict["node"])
    Node = Tree("data_types", Children)
    data_types_output["node"] = Node
    data_types_output["index"] = data_types_tail_dict["index"]
    return data_types_output


def data_types_tail(j):
    data_types_tail_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        if (Token_Type['token_type'] == Token_type.Comma):
            out1 = Match(Token_type.Comma, j)
            Children.append(out1['node'])
            data_type_dict = data_type(out1["index"])
            Children.append(data_type_dict["node"])
            data_types_tail_dict = data_types_tail(data_type_dict["index"])
            Children.append(data_types_tail_dict['node'])
            Node = Tree('data_types_tail', Children)
            data_types_tail_output['node'] = Node
            data_types_tail_output['index'] = data_types_tail_dict['index']
            return data_types_tail_output
        else:
            Node = Tree('data_types_tail', [])
            data_types_tail_output['node'] = Node
            data_types_tail_output['index'] = j
            return data_types_tail_output
    else:
        # Node = Tree('data_types_tail', [])
        # data_types_tail_output['node'] = Node
        data_types_tail_output['node'] = ["error"]
        data_types_tail_output['index'] = j
        return data_types_tail_output


def data_type(j):
    data_type_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        out1 = Match(Token_type.Data_type, j)
        Children.append(out1['node'])
        Node = Tree('data_type', Children)
        data_type_output['node'] = Node
        data_type_output['index'] = out1['index']
        return data_type_output
    else:
        errors.append(
            "Syntax error: Expected a valid datatype but reached end of input ")
        # Node = Tree('data_type', [])
        # data_type_output['node'] = Node
        data_type_output['node'] = ["error"]
        data_type_output['index'] = j
        return data_type_output


def clauses(j):
    clauses_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        if (Token_Type['token_type'] != Token_type.Goal):
            clause_dict = clause(j)
            Children.append(clause_dict["node"])
            clauses_dict = clauses(clause_dict["index"])
            Children.append(clauses_dict['node'])
            Node = Tree("clauses", Children)
            clauses_output["node"] = Node
            clauses_output["index"] = clauses_dict["index"]
            return clauses_output
        else:
            Node = Tree('clauses', [])
            clauses_output['node'] = Node
            clauses_output['index'] = j
            return clauses_output

    else:

        # Node = Tree('clauses', [])
        # clauses_output['node'] = Node
        clauses_output['node'] = ['error']
        clauses_output['index'] = j
        return clauses_output


def clause(j):
    clause_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        for x in range(j, len(Tokens)):
            TokenType = Tokens[x].to_dict()
            if (TokenType['token_type'] == Token_type.Head or TokenType['token_type'] == Token_type.Period):
                break
        if (TokenType['token_type'] != Token_type.Head):
            fact_dict = fact(j)
            Children.append(fact_dict['node'])
            out1 = Match(Token_type.Period, fact_dict['index'])
            Children.append(out1['node'])
            Node = Tree('clause', Children)
            clause_output['node'] = Node
            clause_output['index'] = out1['index']
            return clause_output
        elif (TokenType['token_type'] == Token_type.Head):
            rules_dict = rule(j)
            Children.append(rules_dict['node'])
            out1 = Match(Token_type.Period, rules_dict['index'])
            Children.append(out1['node'])
            Node = Tree('clause', Children)
            clause_output['node'] = Node
            clause_output['index'] = out1['index']
            return clause_output
        else:

            Node = Tree('clause', [])
            clause_output['node'] = Node
            clause_output['index'] = j
            return clause_output
    else:
        # Node = Tree('clauses', [])
        # clause_output['node'] = Node
        clause_output['node'] = ['error']
        clause_output['index'] = j
        return clause_output


def fact(j):
    fact_output = dict()
    Children = []
    out1 = Match(Token_type.Predicate_name, j)
    Children.append(out1["node"])
    fact_output["index"] = out1["index"]
    if (out1["index"] < len(Tokens)):
        Token_Type = Tokens[out1["index"]].to_dict()
        if (Token_Type['token_type'] == Token_type.LeftParenthesis):
            out2 = Match(Token_type.LeftParenthesis, out1["index"])
            Children.append(out2["node"])
            parameters_values_dict = parameters_values(out2["index"])
            Children.append(parameters_values_dict["node"])
            out3 = Match(Token_type.RightParenthesis,
                         parameters_values_dict["index"])
            Children.append(out3["node"])
            fact_output["index"] = out3["index"]
            Node = Tree("fact", Children)
            fact_output["node"] = Node
            return fact_output

        else:
            Node = Tree("fact", Children)
            fact_output["node"] = Node
            fact_output['index'] = out1['index']
            return fact_output
    else:
        Node = Tree("fact", Children)
        fact_output["node"] = Node
        fact_output['index'] = out1['index']
        return fact_output


def parameters_values(j):
    parameters_values_output = dict()
    Children = []
    parameter_value_dict = parameter_value(j)
    Children.append(parameter_value_dict["node"])
    parameters_value_tail_dict = parameters_values_tail(
        parameter_value_dict["index"])
    # parameters_values_tail_dict = parameters_values_tail(parameter_value_dict["index"])
    Children.append(parameters_value_tail_dict['node'])
    Node = Tree('parameters_values', Children)
    parameters_values_output['node'] = Node
    parameters_values_output['index'] = parameters_value_tail_dict['index']
    return parameters_values_output


def parameters_values_tail(j):
    parameters_values_tail_output = dict()
    if (j < len(Tokens)):
        Children = []
        Token_Type = Tokens[j].to_dict()
        if (Token_Type['token_type'] == Token_type.Comma):
            out1 = Match(Token_type.Comma, j)
            Children.append(out1['node'])
            parameter_values_dict = parameter_value(out1['index'])
            Children.append(parameter_values_dict['node'])
            parameter_values_tail_dict = parameters_values_tail(
                parameter_values_dict["index"])
            Children.append(parameter_values_tail_dict['node'])
            Node = Tree('parameters_values_tail', Children)
            parameters_values_tail_output['node'] = Node
            parameters_values_tail_output['index'] = parameter_values_tail_dict['index']
            return parameters_values_tail_output
        else:
            Node = Tree('parameters_values_tail', [])
            parameters_values_tail_output['node'] = Node
            parameters_values_tail_output['index'] = j
            return parameters_values_tail_output
    else:
        # Node = Tree('parameters_values_tail', [])
        # parameters_values_tail_output['node'] = Node
        parameters_values_tail_output['node'] = ['error']
        parameters_values_tail_output['index'] = j
        return parameters_values_tail_output


def parameter_value(j):
    parameter_value_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        if (Token_Type["token_type"] == Token_type.String):
            out1 = Match(Token_type.String, j)
            Children.append(out1['node'])
            Node = Tree('parameter_value', Children)
            parameter_value_output['node'] = Node
            parameter_value_output['index'] = out1['index']
            return parameter_value_output
        elif (Token_Type['token_type'] == Token_type.Symbol):
            out2 = Match(Token_type.Symbol, j)
            Children.append(out2['node'])
            Node = Tree('parameter_value', Children)
            parameter_value_output['node'] = Node
            parameter_value_output['index'] = out2['index']
            return parameter_value_output
        elif (Token_Type["token_type"] == Token_type.Integer):
            out3 = Match(Token_type.Integer, j)
            Children.append(out3['node'])
            Node = Tree('parameter_value', Children)
            parameter_value_output['node'] = Node
            parameter_value_output['index'] = out3['index']
            return parameter_value_output
        elif (Token_Type['token_type'] == Token_type.Real):
            out5 = Match(Token_type.Real, j)
            Children.append(out5['node'])
            Node = Tree('parameter_value', Children)
            parameter_value_output['node'] = Node
            parameter_value_output['index'] = out5['index']
            return parameter_value_output
        elif (Token_Type['token_type'] == Token_type.Char):
            out6 = Match(Token_type.Char, j)
            Children.append(out6['node'])
            Node = Tree('parameter_value', Children)
            parameter_value_output['node'] = Node
            parameter_value_output['index'] = out6['index']
            return parameter_value_output
        else:
            errors.append(
                "Syntax error: Expected a valid parameter value but found " + Tokens[j].to_dict()['Lex'])
            Node = Tree('parameter_value', Children)
            parameter_value_output['node'] = Node
            parameter_value_output['index'] = j
            return parameter_value_output

    else:
        errors.append(
            "Syntax error: Expected a valid parameter value but reached end of input ")
        # Node = Tree('data_type', [])
        # data_type_output['node'] = Node
        parameter_value_output['node'] = ["error"]
        parameter_value_output['index'] = j
        return parameter_value_output


def rule(j):
    rule_output = dict()
    Children = []
    print(j, "INSIDE RULE")
    head_dict = head(j)
    Children.append(head_dict['node'])
    out1 = Match(Token_type.Head, head_dict['index'])
    Children.append(out1['node'])
    body_dict = body(out1['index'])
    Children.append(body_dict['node'])
    Node = Tree('rule', Children)
    rule_output['node'] = Node
    rule_output['index'] = body_dict['index']
    return rule_output


def head(j):
    head_output = dict()
    Children = []
    out1 = Match(Token_type.Predicate_name, j)
    Children.append(out1["node"])
    head_output["index"] = out1["index"]
    if (out1["index"] < len(Tokens)):
        Token_Type = Tokens[out1["index"]].to_dict()
        if (Token_Type['token_type'] == Token_type.LeftParenthesis):
            out2 = Match(Token_type.LeftParenthesis, out1["index"])
            Children.append(out2['node'])
            values_dict = values(out2['index'])
            Children.append(values_dict['node'])
            out3 = Match(Token_type.RightParenthesis, values_dict['index'])
            Children.append(out3['node'])
            head_output['index'] = out3['index']
            Node = Tree("head", Children)
            head_output["node"] = Node
            return head_output
        else:
            Node = Tree("head", Children)
            head_output["node"] = Node
            head_output['index'] = out1['index']
            return head_output

    else:
        Node = Tree("head", Children)
        head_output["node"] = Node
        head_output['index'] = out1['index']
        return head_output


def values(j):
    values_output = dict()
    Children = []
    value_dict = value(j)
    Children.append(value_dict['node'])
    values_tail_dict = values_tail(value_dict['index'])
    Children.append(values_tail_dict['node'])
    Node = Tree('values', Children)
    values_output['node'] = Node
    values_output['index'] = values_tail_dict['index']
    return values_output


def values_tail(j):
    values_tail_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        if (Token_Type['token_type'] == Token_type.Comma):
            out1 = Match(Token_type.Comma, j)
            Children.append(out1['node'])
            value_dict = value(out1['index'])
            Children.append(value_dict['node'])
            values_tail_dict = values_tail(value_dict["index"])
            Children.append(values_tail_dict['node'])
            Node = Tree('values_tail', Children)
            values_tail_output['node'] = Node
            values_tail_output['index'] = values_tail_dict['index']
            return values_tail_output

        else:
            Node = Tree('values_tail', [])
            values_tail_output['node'] = Node
            values_tail_output['index'] = j
            return values_tail_output
    else:
        # Node = Tree('variables_tail', [])
        # variables_tail_output['node'] = Node
        values_tail_output['node'] = ['error']
        values_tail_output['index'] = j
        return values_tail_output


def value(j):
    value_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        if (Token_Type["token_type"] == Token_type.Variable):
            out1 = Match(Token_type.Variable, j)
            Children.append(out1['node'])
            Node = Tree('value', Children)
            value_output['node'] = Node
            value_output['index'] = out1['index']
            return value_output
        elif (Token_Type["token_type"] == Token_type.String):
            out2 = Match(Token_type.String, j)
            Children.append(out2['node'])
            Node = Tree('value', Children)
            value_output['node'] = Node
            value_output['index'] = value_output['index']
            return value_output
        elif (Token_Type['token_type'] == Token_type.Symbol):
            out3 = Match(Token_type.Symbol, j)
            Children.append(out3['node'])
            Node = Tree('value', Children)
            value_output['node'] = Node
            value_output['index'] = out3['index']
            return value_output
        elif (Token_Type["token_type"] == Token_type.Integer):
            out4 = Match(Token_type.Integer, j)
            Children.append(out4['node'])
            Node = Tree('value', Children)
            value_output['node'] = Node
            value_output['index'] = out4['index']
            return value_output
        elif (Token_Type['token_type'] == Token_type.Real):
            out4 = Match(Token_type.Real, j)
            Children.append(out4['node'])
            Node = Tree('value', Children)
            value_output['node'] = Node
            value_output['index'] = out4['index']
            return value_output
        elif (Token_Type['token_type'] == Token_type.Char):
            out5 = Match(Token_type.Char, j)
            Children.append(out5['node'])
            Node = Tree('value', Children)
            value_output['node'] = Node
            value_output['index'] = out5['index']
            return value_output
        elif (Token_Type['token_type'] == Token_type.Annonymous_variable):
            out6 = Match(Token_type.Annonymous_variable, j)
            Children.append(out6['node'])
            Node = Tree('value', Children)
            value_output['node'] = Node
            value_output['index'] = out6['index']
            return value_output
        else:
            errors.append(
                "Syntax error: Expected a valid value but found " + Tokens[j].to_dict()['Lex'])
            Node = Tree('value', Children)
            value_output['node'] = Node
            value_output['index'] = j
            return value_output

    else:
        errors.append(
            "Syntax error: Expected a valid value but reached end of input ")
        Node = Tree('value', [])
        value_output['node'] = Node
        value_output['index'] = j
        return value_output


def body(j):
    body_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        for x in range(j, len(Tokens)):
            TokenType = Tokens[x].to_dict()
            print(TokenType['token_type'])
            if (TokenType['token_type'] == Token_type.Less_than or TokenType['token_type'] == Token_type.Less_than_equal
                or TokenType['token_type'] == Token_type.Greater_than or TokenType['token_type'] == Token_type.Greater_than_equal
                or TokenType['token_type'] == Token_type.Equal or TokenType['token_type'] == Token_type.Difference
                or TokenType['token_type'] == Token_type.Plus or TokenType['token_type'] == Token_type.Minus
                or TokenType['token_type'] == Token_type.Multiply or TokenType['token_type'] == Token_type.Divide
                or TokenType['token_type'] == Token_type.Predicate_name or TokenType['token_type'] == Token_type.Readln
                    or TokenType['token_type'] == Token_type.Readint or TokenType['token_type'] == Token_type.Write):
                break
        if (Token_Type['token_type'] == Token_type.Predicate_name):
            out1 = Match(Token_type.Predicate_name, j)
            Children.append(out1["node"])
            if (out1["index"] < len(Tokens)):
                Token_Type = Tokens[out1["index"]].to_dict()
                if (Token_Type['token_type'] == Token_type.LeftParenthesis):
                    out2 = Match(Token_type.LeftParenthesis, out1["index"])
                    Children.append(out2['node'])
                    values_dict = values(out2['index'])
                    Children.append(values_dict['node'])
                    out3 = Match(Token_type.RightParenthesis,
                                 values_dict['index'])
                    Children.append(out3['node'])
                    body_tail_dict = body_tail(out3["index"])
                    Children.append(body_tail_dict["node"])
                else:
                    body_tail_dict = body_tail(out1["index"])
                    Children.append(body_tail_dict["node"])

        elif (TokenType['token_type'] == Token_type.Less_than or TokenType['token_type'] == Token_type.Less_than_equal
              or TokenType['token_type'] == Token_type.Greater_than or TokenType['token_type'] == Token_type.Greater_than_equal
              or TokenType['token_type'] == Token_type.Equal or TokenType['token_type'] == Token_type.Difference
              or TokenType['token_type'] == Token_type.Plus or TokenType['token_type'] == Token_type.Minus
              or TokenType['token_type'] == Token_type.Multiply or TokenType['token_type'] == Token_type.Divide):
            # elif (Token_Type['token_type'] == Token_type.Variable):
            comparison_dict = comparison(j)
            Children.append(comparison_dict["node"])
            body_tail_dict = body_tail(comparison_dict["index"])
            Children.append(body_tail_dict["node"])
        elif (Token_Type['token_type'] == Token_type.Readln or Token_Type['token_type'] == Token_type.Readint
              or Token_Type['token_type'] == Token_type.Readchar):
            input_predicate_dict = input_predicate(j)
            Children.append(input_predicate_dict["node"])
            body_tail_dict = body_tail(input_predicate_dict["index"])
            Children.append(body_tail_dict["node"])
        elif (Token_Type['token_type'] == Token_type.Write):
            output_predicate_dict = output_predicate(j)
            Children.append(output_predicate_dict["node"])
            body_tail_dict = body_tail(output_predicate_dict["index"])
            Children.append(body_tail_dict["node"])
        elif (Token_Type['token_type'] == Token_type.Exclamation or Token_Type['token_type'] == Token_type.nl):
            output = Match(Token_Type['token_type'], j)
            Children.append(output["node"])
            body_tail_dict = body_tail(output["index"])
            Children.append(body_tail_dict["node"])

        else:
            Node = Tree('body', [])
            body_output['node'] = Node
            body_output['index'] = j
            return body_output
    else:
        # Node = Tree('body', [])
        # body_output['node'] = Node
        body_output['node'] = ['error']
        body_output['index'] = j
        return body_output

    Node = Tree('body', Children)
    body_output['node'] = Node
    # body_output['index'] = out3['index']
    body_output['index'] = body_tail_dict['index']
    return body_output


def body_tail(j):
    body_tail_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        if (Token_Type['token_type'] == Token_type.Comma or Token_Type['token_type'] == Token_type.Semicolon):
            out1 = Match(Token_Type['token_type'], j)
            Children.append(out1['node'])
            body_dict = body(out1['index'])
            Children.append(body_dict['node'])
            Node = Tree('body_tail', Children)
            body_tail_output['node'] = Node
            body_tail_output['index'] = body_dict['index']
            return body_tail_output
        else:
            Node = Tree('body_tail', [])
            body_tail_output['node'] = Node
            body_tail_output['index'] = j
            return body_tail_output
    else:
        # Node = Tree('body_tail', [])
        # body_tail_output['node'] = Node
        body_tail_output['node'] = ['error']
        body_tail_output['index'] = j
        return body_tail_output


def comparison(j):
    comparison_output = dict()
    Children = []
    expression_dict = expression(j)
    Children.append(expression_dict["node"])
    comparator_dict = comparator(expression_dict["index"])
    Children.append(comparator_dict["node"])
    expression2_dict = expression(comparator_dict["index"])
    Children.append(expression2_dict["node"])
    Node = Tree('comparison', Children)
    comparison_output['node'] = Node
    comparison_output['index'] = expression2_dict['index']
    return comparison_output


def expression(j):
    expression_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        if (Token_Type['token_type'] == Token_type.Variable or Token_Type['token_type'] == Token_type.Integer or Token_Type['token_type'] == Token_type.Real):
            out1 = Match(Token_Type['token_type'], j)
            Children.append(out1["node"])
            expression_tail_dict = expression_tail(out1["index"])
            Children.append(expression_tail_dict["node"])
        elif (Token_Type['token_type'] == Token_type.LeftParenthesis):
            out1 = Match(Token_type.LeftParenthesis, j)
            Children.append(out1["node"])
            expression_dict = expression(out1["index"])
            Children.append(expression_dict["node"])
            out2 = Match(Token_type.RightParenthesis, expression_dict["index"])
            Children.append(out2["node"])
            expression_tail_dict = expression_tail(out2["index"])
            Children.append(expression_tail_dict["node"])
        else:
            Node = Tree('expression', [])
            expression_output['node'] = Node
            expression_output['index'] = j
            return expression_output
    else:
        # Node = Tree('expression', [])
        # expression_output['node'] = Node
        expression_output['node'] = ['error']
        expression_output['index'] = j
        return expression_output

    Node = Tree('expression', Children)
    expression_output['node'] = Node
    expression_output['index'] = expression_tail_dict['index']
    return expression_output


def expression_tail(j):
    expression_tail_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        if (Token_Type['token_type'] == Token_type.Plus or Token_Type['token_type'] == Token_type.Minus
                or Token_Type['token_type'] == Token_type.Multiply or Token_Type['token_type'] == Token_type.Divide):
            operator_dict = operator(j)
            Children.append(operator_dict["node"])
            expression_dict = expression(operator_dict["index"])
            Children.append(expression_dict["node"])
            expression_tail_dict = expression_tail(expression_dict["index"])
            Children.append(expression_tail_dict["node"])
            Node = Tree('expression_tail', Children)
            expression_tail_output['node'] = Node
            expression_tail_output['index'] = expression_tail_dict['index']
            return expression_tail_output
        else:
            Node = Tree('expression_tail', [])
            expression_tail_output['node'] = Node
            expression_tail_output['index'] = j
            return expression_tail_output
    else:
        # Node = Tree('expression_tail', [])
        # expression_tail_output['node'] = Node
        expression_tail_output['node'] = ['error']
        expression_tail_output['index'] = j
        return expression_tail_output


def operator(j):
    operator_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        if (Token_Type['token_type'] == Token_type.Plus or Token_Type['token_type'] == Token_type.Minus
                or Token_Type['token_type'] == Token_type.Multiply or Token_Type['token_type'] == Token_type.Divide):
            operator1 = Match(Token_Type['token_type'], j)
            Children.append(operator1["node"])
            Node = Tree('operator', Children)
            operator_output['node'] = Node
            operator_output['index'] = operator1["index"]
            return operator_output
        else:
            Node = Tree('operator', [])
            operator_output['node'] = Node
            operator_output['index'] = j
            return operator_output
    else:
        Node = Tree('operator', [])
        operator_output['node'] = Node
        operator_output['index'] = j
        return operator_output


def comparator(j):
    comparator_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        if (Token_Type['token_type'] == Token_type.Less_than or Token_Type['token_type'] == Token_type.Less_than_equal
            or Token_Type['token_type'] == Token_type.Greater_than or Token_Type[
            'token_type'] == Token_type.Greater_than_equal
                or Token_Type['token_type'] == Token_type.Equal or Token_Type['token_type'] == Token_type.Difference):
            out1 = Match(Token_Type['token_type'], j)
            Children.append(out1["node"])
            Node = Tree('comparator', Children)
            comparator_output['node'] = Node
            comparator_output['index'] = out1["index"]
            return comparator_output
        else:
            Node = Tree('operator', [])
            comparator_output['node'] = Node
            comparator_output['index'] = j
            return comparator_output

    else:
        Node = Tree('operator', [])
        comparator_output['node'] = Node
        comparator_output['index'] = j
        return comparator_output


def input_predicate(j):
    input_predicate_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        if (Token_Type['token_type'] == Token_type.Readln or Token_Type['token_type'] == Token_type.Readint or
                Token_Type['token_type'] == Token_type.Readchar):
            out1 = Match(Token_Type['token_type'], j)
            Children.append(out1["node"])
            out2 = Match(Token_type.LeftParenthesis, out1["index"])
            Children.append(out2["node"])
            out3 = Match(Token_type.Variable, out2["index"])
            Children.append(out3["node"])
            out4 = Match(Token_type.RightParenthesis, out3["index"])
            Children.append(out4["node"])
            Node = Tree('input_predicate', Children)
            input_predicate_output['node'] = Node
            input_predicate_output['index'] = out4["index"]
            return input_predicate_output
        else:
            Node = Tree('operator', [])
            input_predicate_output['node'] = Node
            input_predicate_output['index'] = j
            return input_predicate_output

    else:
        Node = Tree('operator', [])
        input_predicate_output['node'] = Node
        input_predicate_output['index'] = j
        return input_predicate_output


def output_predicate(j):
    output_predicate_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        if (Token_Type['token_type'] == Token_type.Write):
            out1 = Match(Token_type.Write, j)
            Children.append(out1["node"])
            out2 = Match(Token_type.LeftParenthesis, out1["index"])
            Children.append(out2["node"])
            outputs_dict = outputs(out2["index"])
            # outputs_dict =  Match(Token_type.String, out2["index"])
            Children.append(outputs_dict["node"])
            out3 = Match(Token_type.RightParenthesis, outputs_dict["index"])
            Children.append(out3["node"])
            Node = Tree('output_predicate', Children)
            output_predicate_output['node'] = Node
            output_predicate_output['index'] = out3["index"]
            return output_predicate_output
        else:
            Node = Tree('operator', [])
            output_predicate_output['node'] = Node
            output_predicate_output['index'] = j
            return output_predicate_output
    else:
        Node = Tree('operator', [])
        output_predicate_output['node'] = Node
        output_predicate_output['node'] = ['error']
        output_predicate_output['index'] = j
        return output_predicate_output


def outputs(j):
    outputs_output = dict()
    Children = []
    output_dict = output(j)
    Children.append(output_dict["node"])
    outputs_tail_dict = outputs_tail(output_dict["index"])
    Children.append(outputs_tail_dict["node"])
    Node = Tree('outputs', Children)
    outputs_output['node'] = Node
    outputs_output['index'] = outputs_tail_dict['index']
    return outputs_output


def outputs_tail(j):
    outputs_tail_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        if (Token_Type['token_type'] == Token_type.Comma):
            out1 = Match(Token_type.Comma, j)
            Children.append(out1["node"])
            output_dict = output(out1["index"])
            Children.append(output_dict["node"])
            outputs_tail_dict = outputs_tail(output_dict["index"])
            Children.append(outputs_tail_dict["node"])
            Node = Tree('outputs_tail', Children)
            outputs_tail_output['node'] = Node
            outputs_tail_output['index'] = outputs_tail_dict['index']
            return outputs_tail_output
        else:
            Node = Tree('outputs_tail', [])
            outputs_tail_output['node'] = Node
            outputs_tail_output['index'] = j
            return outputs_tail_output
    else:
        # Node = Tree('outputs_tail', [])
        # outputs_tail_output['node'] = Node
        outputs_tail_output['node'] = ['error']
        outputs_tail_output['index'] = j
        return outputs_tail_output


def output(j):
    output_output = dict()
    Children = []
    if (j < len(Tokens)):
        Token_Type = Tokens[j].to_dict()
        if (Token_Type['token_type'] == Token_type.Variable):
            out1 = Match(Token_type.Variable, j)
            Children.append(out1['node'])
            Node = Tree('output', Children)
            output_output['node'] = Node
            output_output['index'] = out1['index']
            return output_output

        elif (Token_Type['token_type'] == Token_type.String):
            out2 = Match(Token_type.String, j)
            Children.append(out2["node"])
            Node = Tree('output', Children)
            output_output['node'] = Node
            output_output['index'] = out2['index']
            return output_output

        elif (Token_Type['token_type'] == Token_type.Integer):
            out3 = Match(Token_type.Integer, j)
            Children.append(out3['node'])
            Node = Tree('value', Children)
            Node = Tree('output', Children)
            output_output['node'] = Node
            output_output['index'] = out3['index']
            return output_output
        else:
            errors.append(
                "Syntax error: Expected a valid output but found " + Tokens[j].to_dict()['Lex'])
            Node = Tree('output', Children)
            output_output['node'] = Node
            output_output['index'] = j
            return output_output

    else:
        errors.append(
            "Syntax error: Expected a valid output but reached end of input ")
        # Node = Tree('output', [])
        # output_output['node'] = Node
        output_output['node'] = ['error']
        output_output['index'] = j
        return output_output


def goal(j):

    goal_output = dict()
    Children = []
    out1 = Match(Token_type.Predicate_name, j)
    Children.append(out1["node"])
    goal_output["index"] = out1["index"]
    if (out1["index"] < len(Tokens)):
        Token_Type = Tokens[out1["index"]].to_dict()
        if (Token_Type['token_type'] == Token_type.LeftParenthesis):
            out2 = Match(Token_type.LeftParenthesis, out1["index"])
            Children.append(out2["node"])
            values_dict = values(out2["index"])
            Children.append(values_dict["node"])
            out3 = Match(Token_type.RightParenthesis, values_dict["index"])
            Children.append(out3["node"])
            out4 = Match(Token_type.Period, out3['index'])
            Children.append(out4["node"])
            Node = Tree('Goal', Children)
            goal_output["node"] = Node
            goal_output["index"] = out4["index"]
            return goal_output
        elif (Token_Type['token_type'] == Token_type.Period):
            out2 = Match(Token_type.Period, out1['index'])
            Children.append(out2["node"])
            Node = Tree('Goal', Children)
            goal_output['node'] = Node
            goal_output['index'] = out2['index']
            return goal_output
        else:
            Node = Tree('Goal', Children)
            goal_output['node'] = Node
            goal_output['index'] = out1['index']
            return goal_output
    else:
        Node = Tree('Goal', Children)
        goal_output['node'] = Node
        goal_output['index'] = out1['index']
        return goal_output


def Match(expected_token_type, j):
    output = dict()
    if (j < len(Tokens)):
        current_token = Tokens[j].to_dict()
        if (current_token['token_type'] == expected_token_type):
            j += 1
            output["node"] = [current_token['Lex']]
            output["index"] = j
            return output
        else:
            output["node"] = ["error"]
            output["index"] = j + 1
            errors.append("Syntax error: Expected " +
                          expected_token_type.name + " but found " + current_token['Lex'])
            j += 1  # Skip the current token

            # Perform error skipping until a valid token is found
            while j < len(Tokens):
                current_token = Tokens[j].to_dict()
                if current_token['token_type'] == expected_token_type:
                    output["node"] = [current_token['Lex']]
                    output["index"] = j + 1
                    errors.append("Recovered from syntax error")
                    return output
                j += 1

            errors.append("Syntax error: Reached end of input")
            return output
    else:
        output["node"] = ["error"]
        output["index"] = j + 1
        errors.append("Syntax error: Expected " +
                      expected_token_type.name + " but reached end of input")
    return output


def generate_static_dfa_diagram():
    x1 = text.get("1.0", tk.END).strip()
    scan_prolog(x1)

    # Create a Digraph object
    dot = Digraph(comment='DFA for Valid Tokens', graph_attr={'fontname': 'SANS , bold', 'fontsize': '12', 'rankdir': 'TB'},
                  node_attr={'fontname': 'SANS , bold', 'fontsize': '12',
                             'style': 'filled', 'fillcolor': 'lightblue'},
                  edge_attr={'fontname': 'SANS , bold', 'fontsize': '14', 'color': 'red'})

    # Keep track of visited nodes and edges
    visited_nodes = set()
    visited_edges = set()

    # Add the first node with an arrow indicating the initial state
    first_node_label = str(Tokens[0].token_type.name)
    dot.node(first_node_label, label=first_node_label,
             shape='ellipse', peripheries='1')
    # Set the shape to 'none' for the arrow node
    dot.attr('node', shape='none', fillcolor='none')
    dot.node('start', label='', shape='none')
    dot.edge('start', first_node_label, label='',
             arrowhead='normal', dir='forward')
    # Reset the shape to 'ellipse' for the remaining nodes
    dot.attr('node', shape='ellipse', fillcolor='lightblue')

    # Add nodes and edges for valid tokens
    for token in Tokens:
        if token.token_type == Token_type.Error:
            continue

        node_label = str(token.token_type.name)

        # Add the node to the graph if it hasn't been visited before
        if node_label not in visited_nodes:
            dot.node(node_label, label=node_label, fillcolor='lightblue')
            visited_nodes.add(node_label)

        # Add the edge from the current node to the next node
        next_token_index = Tokens.index(token) + 1
        if next_token_index < len(Tokens):
            next_token = Tokens[next_token_index]
            if next_token.token_type == Token_type.Error:
                continue

            next_node_label = str(next_token.token_type.name)
            edge_label = token.lex  # Set the edge label as the token lex

            # Build a unique identifier for the edge
            edge_identifier = f"{node_label}-{next_node_label}-{edge_label}"

            # Add the edge if it hasn't been visited before
            if edge_identifier not in visited_edges:
                dot.edge(node_label, next_node_label, label=edge_label)
                visited_edges.add(edge_identifier)

    # Add the double circle to the last node
    last_node_label = str(Tokens[-1].token_type.name)
    dot.node(last_node_label, shape='ellipse',
             peripheries='2', fillcolor='lightgreen')

    # Save the diagram as an image
    dot.render('dfa_diagram', format='jpeg',
               view=True, cleanup=True, engine='dot')
    pass


def generate_animated_dfa_diagram():
    x2 = text.get("1.0", tk.END).strip()
    scan_prolog(x2)

    # Create a new directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for token in Tokens:
        if token.token_type == Token_type.Error:
            continue
        G.add_node(str(token.token_type.name))

    # Add edges to the graph
    for i in range(len(Tokens) - 1):
        current_token = Tokens[i]
        next_token = Tokens[i + 1]
        if current_token.token_type == Token_type.Error or next_token.token_type == Token_type.Error:
            continue
        G.add_edge(str(current_token.token_type.name),
                   str(next_token.token_type.name))

    # Add the last node as a double peripheral node
    last_node = str(Tokens[-1].token_type.name)
    G.add_node(last_node)
    G.add_edge(last_node, last_node)

    # Initialize the plot
    fig, ax = plt.subplots()

    # Set the layout of the graph using shell_layout
    pos = nx.shell_layout(G)

    # Animation loop
    for i, token in enumerate(Tokens):
        if token.token_type == Token_type.Error:
            continue

        node_label = str(token.token_type.name)

        # Update the node and edge colors for the current step
        node_colors = ['red' if n == node_label else 'blue' for n in G.nodes]
        edge_colors = ['red' if u == node_label and v == str(Tokens[i + 1].token_type.name) else 'gray'
                       for u, v in G.edges]

        # Clear the plot and draw the updated graph
        ax.clear()
        nx.draw_networkx(G, pos, node_color=node_colors, edge_color=edge_colors, ax=ax,
                         with_labels=True, node_size=600, font_size=11, font_family="sans-serif,bold")

        # Draw edge labels
        edge_labels = {(u, v): Tokens[i + 1].lex if u == node_label and v == str(Tokens[i + 1].token_type.name) else ''
                       for u, v in G.edges}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=10, font_family="sans-serif,bold", ax=ax)

        # Set the title for the current step
        if i == len(Tokens) - 2:
            node_colors[-1] = 'green'  # Change color of last node to green
            ax.set_title(f"Final Node: {node_label}")
        else:
            ax.set_title(f"Q {i}: {node_label}")

        # Pause for a short duration to create an animation effect
        plt.pause(1.5)

        # Terminate animation when the window is closed
        if not plt.fignum_exists(fig.number):
            break

    # Show the final plot
    plt.show()
    pass


def generate_animated_dfa_diagram_2():
    x3 = text.get("1.0", tk.END).strip()
    scan_prolog(x3)

    # Create a new directed graph
    G = nx.DiGraph()

    # Initialize the plot
    fig, ax = plt.subplots()

    # Set the initial layout of the graph
    pos = {}

    # Define the update function for the animation
    def update(frame):
        node_label = str(Tokens[frame].token_type.name)

        # Add the current node to the graph if it doesn't exist yet
        if node_label not in G.nodes:
            G.add_node(node_label)
            pos[node_label] = (frame, 0)

        # Add the next node to the graph if it doesn't exist yet
        if frame < len(Tokens) - 1:
            next_node_label = str(Tokens[frame + 1].token_type.name)
            if next_node_label not in G.nodes:
                G.add_node(next_node_label)
                pos[next_node_label] = (frame + 1, random.uniform(-0.5, 0.5))

            # Add the edge between the current node and the next node
            if not G.has_edge(node_label, next_node_label):
                G.add_edge(node_label, next_node_label, label='')

        # Update the node and edge colors for the current step
        node_colors = ['red' if n == node_label else 'blue' for n in G.nodes]
        edge_colors = ['red' if u == node_label and v == str(Tokens[frame + 1].token_type.name) else 'gray'
                       for u, v in G.edges]

        # Clear the plot and draw the updated graph
        ax.clear()
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, ax=ax, node_size=1000)
        nx.draw_networkx_edges(
            G, pos, edge_color=edge_colors, ax=ax, width=1, arrowsize=25)
        nx.draw_networkx_labels(
            G, pos, font_size=10, font_family="sans-serif", ax=ax, font_weight='bold')

        # Set the label for the selected edge
        selected_edge = (node_label, str(Tokens[frame + 1].token_type.name))
        edge_labels = {edge: Tokens[frame].lex if edge ==
                       selected_edge else '' for edge in G.edges}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=14, font_family="sans-serif", ax=ax)

        # Set the title for the current step
        ax.set_title(f"Q {frame}: {node_label}")

    # Create and run the animation
    anim = FuncAnimation(fig, update, frames=len(
        Tokens) - 1, interval=1500, repeat=False)

    # Show the final plot
    plt.show()
    pass


def grouping(words):
    group = {}
    i = 0
    words = sorted(words)
    while i < len(words):
        matched = 0
        for x in range(i + 1, len(words)):
            didMatch = False
            inner_group = []
            word1 = words[i] if len(words[i]) >= len(words[x]) else words[x]
            word2 = words[i] if len(words[i]) < len(words[x]) else words[x]
            print(word1, " - ", word2)
            for k in range(0, len(word1)):
                if k < len(word2) and word1[k] == word2[k]:
                    if not didMatch:
                        didMatch = True
                        matched = matched + 1
                    inner_group.append(word1[k])
                else:
                    if not didMatch:
                        break
                    inner_group = ''.join(inner_group)
                    # for key in group:
                    #     if key.startswith(inner_group) and key != inner_group:
                    #         key_data = group[key]
                    #         postfix = key[len(inner_group):]
                    #         for p in range(0, len(key_data)):
                    #             key_data[p] = postfix + key_data[p]
                    #         group.pop(key)
                    #         group[inner_group] = []
                    #         group[inner_group].extend(key_data)
                    #         break
                    if inner_group not in group:
                        group[inner_group] = [word1[k:]]
                    group[inner_group].append(
                        word2[k:] if group[inner_group].count(word2[k:]) == 0 else word1[k:])
                    break
        if matched == 0:
            group[words[i]] = []
            i = i + 1
        else:
            i = i + matched + 1
    return group


def generate_dfa_language():
    words = ["predicates", "clauses", "goal", "readln", "readint",
             "readchar", "write", "writeint", "string", "integer", "symbol", "char"]
    # Create a Digraph object
    dot = Digraph(comment='DFA for Valid Tokens', graph_attr={'fontname': 'SANS , bold', 'fontsize': '12', 'rankdir': 'TB'},
                  node_attr={'fontname': 'SANS , bold', 'fontsize': '12',
                             'style': 'filled', 'fillcolor': 'lightblue'},
                  edge_attr={'fontname': 'SANS , bold', 'fontsize': '14', 'color': 'red'})

    data = grouping(words)
    print(data)

    dot.node("INDENTIFIER", label="INDENTIFIER", fillcolor='lightblue')
    dot.edge("INDENTIFIER", "INDENTIFIER", label="[a-z][0-9]_")
    index = 0
    parent_index = 0
    counter = 0
    start_letters = []
    parents_nodes = []
    draw_real = False
    for key in data:
        start_letters.append(key[0])
    for key in data:
        parent_data = data[key]
        # key = key[1:]
        counter += 1
        for i in range(1, len(key)):
            if i == 1:
                parents_nodes.append(counter)
            node_label = "q" + str(counter)
            counter += 1
            dot.node(node_label, label=node_label, fillcolor='lightblue')
            next_node_label = "q" + str(counter)
            edge_label = key[i]
            if draw_real:
                dot.edge("q" + str(counter-2),
                         next_node_label, label=edge_label)
                iel = "[a-z]~[ld][0-9]_"
                dot.edge("q" + str(counter-2), "INDENTIFIER", label=iel)
                draw_real = False
            else:
                dot.edge(node_label, next_node_label, label=edge_label)
                id_edge_label = "[a-z]~" + key[i] + "[0-9]_"
            dot.edge(node_label, "INDENTIFIER", label=id_edge_label)
            parent_index = counter

            if key == 'read' and key[i] == 'a':
                nl = "q" + str(counter+1)
                print("q" + str(counter), " -> ", nl)
                dot.node(nl, label=nl, fillcolor='lightblue')
                dot.edge("q" + str(counter), nl, label="l")
                draw_real = True
                counter += 1

            if i == len(key) - 1 and (len(parent_data) == 0 or '' in parent_data):
                if not '' in parent_data:
                    id_edge_label = "[a-z][0-9]_"
                    dot.edge(next_node_label, "INDENTIFIER",
                             label=id_edge_label)
                dot.node(next_node_label, shape='ellipse',
                         peripheries='2')

        if len(key) <= 1:
            parent_index = counter
            parents_nodes.append(counter)

        child_letters = []
        print(key, " + ", parent_data)
        for k in range(0, len(parent_data)):
            if not parent_data[k]:
                continue
            child_letters.append(parent_data[k][0])

        for k in range(0, len(parent_data)):
            print(parent_data[k])
            for x in range(0, len(parent_data[k])):
                if x == 0:
                    node_label = "q" + str(parent_index + x)
                else:
                    node_label = "q" + str(counter)
                    id_edge_label = "[a-z]~" + parent_data[k][x] + "[0-9]_"
                    dot.edge(node_label, "INDENTIFIER", label=id_edge_label)
                if k == 0 and x == 0:
                    id_edge_label = "[a-z]~[" + \
                        ''.join(child_letters) + "][0-9]_"
                    dot.edge(node_label, "INDENTIFIER", label=id_edge_label)
                counter += 1
                dot.node(node_label, label=node_label, fillcolor='lightblue')
                next_node_label = "q" + str(counter)
                edge_label = parent_data[k][x]
                print("==> ", node_label, " - ",
                      next_node_label, " - ", edge_label)
                dot.edge(node_label, next_node_label, label=edge_label)
                if x == len(parent_data[k]) - 1:
                    dot.node(next_node_label, shape='ellipse',
                             peripheries='2')
                    # Connect last node to Indentifier
                    id_edge_label = "[a-z][0-9]_"
                    dot.edge(next_node_label, "INDENTIFIER",
                             label=id_edge_label)

        index += 1

    node_label = "q0"
    dot.node(node_label, label=node_label, fillcolor='lightblue')
    print(start_letters)
    print(parents_nodes)
    for i in range(0, len(parents_nodes)):
        next_node_label = "q" + str(parents_nodes[i])
        edge_label = start_letters[i]
        dot.edge(node_label, next_node_label, label=edge_label)
    dot.edge(node_label, "INDENTIFIER",
             label="[a-z]~[" + ''.join(start_letters) + "]_")

    node_label = "OPERATORS"
    dot.node(node_label, label=node_label, fillcolor='lightblue')
    dot.edge("q0", node_label, label=":-|:=|>=|<=|<>|[,;.\-+=\*/><]")

    node_label = "ERROR"
    dot.node(node_label, label=node_label, fillcolor='lightblue')
    dot.edge("q0", node_label, label="[0-9]")
    # Save the diagram as an image
    dot.render("all_dfa", format='jpeg',
               cleanup=True, engine='dot', view=True, directory='diagrams')


# words = ["predicates", "clauses", "goal", "readln", "readint",
#          "readchar", "write", "writeint", "string", "integer", "symbol", "char"]
# words = ["readln", "readint", "readchar", "goal", "write"]
# words = ["goals", "predicates"]]


# for word in words:
#     generate_dfa(word)


def generate_dfa_word():

    words = ["predicates", "clauses", "goal", "readln", "readint",
             "readchar", "write", "string", "integer", "symbol", "char", "real"]
    # Create a Digraph object
    for word in words:
        dot = Digraph(comment='DFA for Valid Tokens', graph_attr={'fontname': 'SANS , bold', 'fontsize': '12', 'rankdir': 'TB'},
                      node_attr={'fontname': 'SANS , bold', 'fontsize': '12',
                                 'style': 'filled', 'fillcolor': 'lightblue'},
                      edge_attr={'fontname': 'SANS , bold', 'fontsize': '14', 'color': 'red'})
        dot.node("IDENTIFIER", label="IDENTIFIER", fillcolor='lightblue')
        # Add nodes and edges for valid tokens
        for i in range(0, len(word)):
            alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
            letter_index = alphabet.index(word[i])
            node_label = "q" + str(i)

            dot.node(node_label, label=node_label, fillcolor='lightblue')

            next_node_label = "q" + str(i+1)
            id_edge_label = alphabet[0] + (("-" + alphabet[letter_index-1]) if alphabet[0] != alphabet[letter_index-1] else '') + \
                "|" + alphabet[letter_index+1] + \
                (("-" + alphabet[-1]) if alphabet[letter_index+1]
                 != alphabet[-1] else '')
            edge_label = alphabet[letter_index]
            dot.edge(node_label, next_node_label, label=edge_label)
            dot.edge(node_label, "IDENTIFIER", label=id_edge_label)

        # Add the double circle to the last node
        # last_node_label = str(Tokens[-1].token_type.name)
        last_node = "q" + str(len(word))
        dot.node(last_node, shape='ellipse',
                 peripheries='2', fillcolor='lightgreen')
        dot.edge(last_node, "IDENTIFIER", label="a-z")
        dot.edge("IDENTIFIER", "IDENTIFIER", label="a-z")

        # Save the diagram as an image
        dot.render(word + "_dfa", format='jpeg',
                   cleanup=True, engine='dot', directory='diagrams')

# GUI


def create_token_stream_window():
    df = pd.DataFrame.from_records([t.to_dict() for t in Tokens])
    token_stream_window = tk.Toplevel()
    token_stream_window.title('Token Stream')

    token_stream_table = pt.Table(
        token_stream_window, dataframe=df, showtoolbar=True, showstatusbar=True)
    token_stream_table.show()


def create_error_list_window():
    df = pd.DataFrame(errors)

    error_list_window = tk.Toplevel()
    error_list_window.title('Error List')

    error_list_table = pt.Table(
        error_list_window, dataframe=df, showtoolbar=True, showstatusbar=True)
    error_list_table.show()


def scan_and_parse():
    source_code = text.get("1.0", tk.END).strip()
    tokens = scan_prolog(source_code)
    Node = Parse()
    create_token_stream_window()
    create_error_list_window()
    Node.draw()


# sys.setrecursionlimit(10000)
print(sys.getrecursionlimit())


root = tk.Tk()
root.title('Prolog Compiler')
root.geometry('600x400')  # Set the window size to 600x400 pixels
root.iconify()  # Minimize the window initially

# Use ttk style for a modern look
style = ttk.Style(root)
# Choose one of the available themes (clam, alt, default, classic)
style.theme_use('clam')

# Define custom styles for buttons
style.configure('Blue.TButton', foreground='#8C9FBE', background='#0C134F')
style.configure('LighterBlue.TButton', foreground='#8C9FBE',
                background='#1D267D')  # Style for button1
style.configure('Purple.TButton', foreground='#27374D',
                background='#5C469C')  # Style for button2
style.configure('LighterPurple.TButton', foreground='#27374D',
                background='#D4ADFC')  # Style for button3

# Disable fading effect when hovering over buttons
style.map('Blue.TButton',
          foreground=[('active', 'white'), ('disabled', '#0C134F')],
          background=[('active', '#0C134F'), ('disabled', '#0C134F')])
style.map('LigherBlue.TButton',
          foreground=[('active', 'white'), ('disabled', '#1D267D')],
          background=[('active', '#1D267D'), ('disabled', '#1D267D')])
style.map('Purple.TButton',
          foreground=[('active', 'white'), ('disabled', '#5C469C')],
          background=[('active', '#5C469C'), ('disabled', '#5C469C')])
style.map('LighterPurple.TButton',
          foreground=[('active', 'white'), ('disabled', '#D4ADFC')],
          background=[('active', '#D4ADFC'), ('disabled', '#D4ADFC')])

frame = ttk.Frame(root, padding='20')
frame.pack(fill='both', expand=True)

label1 = ttk.Label(frame, text='Scanner Phase', font=('Helvetica', 14))
label1.pack()

text_frame = ttk.Frame(frame)
text_frame.pack(fill='both', expand=True)

scrollbar = ttk.Scrollbar(text_frame)
scrollbar.pack(side='right', fill='y')

# Adjust the height of the text area
text = tk.Text(text_frame, font=('Helvetica', 10), height=10)
text.pack(fill='both', expand=True)

scrollbar.config(command=text.yview)
text.config(yscrollcommand=scrollbar.set)

button6 = ttk.Button(root, text='Generate DFA Language',
                     command=generate_dfa_language, style='LighterPurple.TButton')
button6.pack(side='bottom', pady=5)

button5 = ttk.Button(root, text='Generate DFA Reserved',
                     command=generate_dfa_word, style='LighterPurple.TButton')
button5.pack(side='bottom', pady=5)

button4 = ttk.Button(root, text='Generate Frame-by-Frame DFA Diagram',
                     command=generate_animated_dfa_diagram_2, style='Purple.TButton')
button4.pack(side='bottom', pady=5)

button3 = ttk.Button(root, text='Generate Animated DFA Diagram',
                     command=generate_animated_dfa_diagram, style='Purple.TButton')
button3.pack(side='bottom', pady=5)

button2 = ttk.Button(root, text='Generate Static DFA Diagram',
                     command=generate_static_dfa_diagram, style='LighterBlue.TButton')
button2.pack(side='bottom', pady=5)

button1 = ttk.Button(
    root, text='Scan', command=scan_and_parse, style='LighterBlue.TButton')
button1.pack(side='bottom', pady=10)

root.deiconify()  # Show the window after adding the buttons

root.mainloop()
