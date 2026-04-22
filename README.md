# 📚 Notebook Video Script Converter

A Python tool that converts Jupyter notebooks into structured **10-minute lecture scripts** using Large Language Models (LLMs).

The system is designed for educators who want to automatically transform teaching notebooks into **spoken lecture content for video production**.

---

## 🚀 Features

- 📓 Parses Jupyter notebooks (`.ipynb`)
- 🧠 Extracts **sections, explanations, and code**
- 🧩 Builds a **lecture outline automatically**
- 🪄 Generates structured teaching scripts using LLMs
- 🧾 Produces ~10-minute spoken lecture scripts
- 🎙️ Optimized for video narration (natural, conversational Spanish output)
- 🔁 Supports batch processing of multiple notebooks
- 🔐 Uses environment variables for secure API key handling

---

## 🏗️ Architecture

The system follows a **section-based teaching pipeline**:

Notebook → Extract structured cells → Group into sections → Build outline → Generate section scripts (with memory) → Final lecture script

---
