#!/usr/bin/env python
from flask import Flask
app = Flask(__name__)

@app.route("/")
def default_response():
    return "\nThis is the default response. No purchase or sale!\n"

@app.route("/purchase_a_sword")
def purchase_sword():
    return "\nSword Purchased!\n"

@app.route("/sell_a_sword")
def sell_sword():
    return "\nSword Purchased!\n"
