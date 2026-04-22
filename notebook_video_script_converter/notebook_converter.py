from openai import OpenAI
import nbformat
from nbformat import read
import re
from dotenv import load_dotenv
import os
import argparse

# Load the environment variables
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# Load the notebook
def load_notebook(path):
    with open(path, "r") as f:
        return nbformat.read(f, as_version=4)

# Extract the cells from the notebook
def extract_cells(nb):
    extracted = []

    for cell in nb.cells:
        if cell.cell_type == "markdown":
            content = cell.source.strip()
            
            # detect headings
            if re.match(r"^#+ ", content):
                extracted.append({
                    "type": "section",
                    "content": content
                })
            else:
                extracted.append({
                    "type": "markdown",
                    "content": content
                })

        elif cell.cell_type == "code" and cell.source.strip():
            extracted.append({
                "type": "code",
                "content": cell.source.strip()
            })

    return extracted

# Format the cells for the LLM
def format_for_llm(cells):
    parts = []

    for c in cells:
        if c["type"] == "section":
            parts.append(f"\nSECTION:\n{c['content']}\n")
        elif c["type"] == "markdown":
            parts.append(f"EXPLANATION:\n{c['content']}")
        elif c["type"] == "code":
            parts.append(f"CODE EXAMPLE:\n{c['content']}")

    return "\n\n".join(parts)

# Group the cells by sections
def group_by_sections(cells):
    sections = []
    current = {"title": "Introduction", "content": []}

    for c in cells:
        if c["type"] == "section":
            if current["content"]:
                sections.append(current)

            title = re.sub(r"^#+ ", "", c["content"]).strip()
            current = {"title": title, "content": []}
        else:
            current["content"].append(c)

    if current["content"]:
        sections.append(current)

    return sections

# Format the section for the LLM
def format_section(section):
    parts = [f"SECTION: {section['title']}"]

    for c in section["content"]:
        if c["type"] == "markdown":
            parts.append(f"EXPLANATION:\n{c['content']}")
        elif c["type"] == "code":
            parts.append(f"CODE:\n{c['content']}")

    return "\n\n".join(parts)

# Chunk the text into smaller chunks
def chunk_text(text, max_chars=4000):
    paragraphs = text.split("\n\n")

    chunks = []
    current = ""

    for p in paragraphs:

        # safety: handle very large paragraphs
        if len(p) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            # hard split large block (rare but safe)
            for i in range(0, len(p), max_chars):
                chunks.append(p[i:i+max_chars])

            continue

        # normal accumulation
        if len(current) + len(p) < max_chars:
            current += p + "\n\n"
        else:
            chunks.append(current.strip())
            current = p + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks

def build_outline(full_text):
    prompt = f"""
    You are a university professor designing a Python lecture.

    Create a structured lecture outline from the content.

    Requirements:
    - Divide into 4–8 logical sections
    - Give each section a clear title
    - Ensure pedagogical order (from simple → complex)
    - No detailed explanations yet, only structure

    Content:
    {full_text}
    """
    return call_llm(prompt)

# Call the LLM
def call_llm(prompt, model="gpt-4.1-mini", temperature=0.3):
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
        )
        return response.output_text
    except Exception as e:
        return f"[ERROR] {e}"

def summarize_with_outline(chunk, outline, context=""):
    prompt = f"""
    You are a university professor preparing lecture material.

    You MUST follow this lecture structure:
    {outline}

    Previous context:
    {context}

    New content:
    {chunk}

    Task:
    - Summarize this chunk
    - Assign it to the correct section in the outline
    - Keep continuity with previous content
    - Focus on teaching clarity and intuition

    Avoid repetition and low-value details.
    """
    return call_llm(prompt)

# Summarize a chunk of content
def summarize_chunk(chunk, context=""):
    prompt = f"""
    You are a university professor preparing lecture material.

    You are building on previous content.

    ### Previous context (important):
    {context}

    ### New content to summarize:
    {chunk}

    ### Task:
    Summarize the new content into structured teaching notes.

    Focus on:
    - Core concepts students must understand
    - How this connects to previous material
    - Key examples
    - Common beginner misunderstandings

    Avoid:
    - Repeating previous explanations unnecessarily
    - Low-value details
    """
    return call_llm(prompt)

# Generate a script from a list of summaries
def generate_script(summaries):
    combined = "\n\n".join(summaries)

    prompt = f"""
    You are a university professor creating a 10-minute lecture script in Spanish.
    The script will be used to create a video lecture for a university course.

    Constraints:
    - Target ~2000 words
    - Do NOT exceed 5000 words
    - Audience: beginner Python students in Spanish
    - Style: spoken, natural, engaging

    Structure:
    1. Introduction (hook + context)
    2. Concept explanations
    3. Code walkthrough (only key parts)
    4. Summary

    Rules:
    - Do NOT read code line-by-line
    - Explain intuition clearly
    - Use simple language
    - Keep it concise but complete
    - Write as if speaking to students
    - Use short sentences
    - Add natural transitions
    - Occasionally include [pause] markers
    - Write in Spanish

    Content:
    {combined}
    """

    return call_llm(prompt)

def generate_section_script(section_text, context=""):
    prompt = f"""
    You are a university professor teaching Python.

    You are writing ONE section of a lecture.

    ### Previous context (important):
    {context}

    ### Section content:
    {section_text}

    ### Task:
    - Explain this section clearly
    - Keep it as a spoken lecture (1–2 minutes)
    - Use intuition, not just definitions
    - Include code explanation only when necessary
    - Ensure continuity with previous sections

    Style:
    - Natural spoken language
    - Short sentences
    - Occasional [pause]
    """
    return call_llm(prompt)

# Trim the script to the desired length
def trim_to_length(text, max_words=5000):
    words = text.split()
    if len(words) <= max_words:
        return text
    
    trimmed = " ".join(words[:max_words])
    
    # try to end at last full stop
    if "." in trimmed:
        trimmed = trimmed.rsplit(".", 1)[0] + "."
    
    return trimmed

# Convert a notebook to a script
def notebook_to_script(path):
    nb = load_notebook(path)
    cells = extract_cells(nb)
    text = format_for_llm(cells)
    
    chunks = chunk_text(text)

    print(f"Chunks: {len(chunks)}")
    
    # 1. GLOBAL OUTLINE FIRST
    outline = build_outline(text)

    summaries = []
    context = ""

    # 2. SECTION-AWARE SUMMARIES
    for chunk in chunks:
        summary = summarize_with_outline(chunk, outline, context)
        summaries.append(summary)

        # compressed memory
        context = call_llm(
            f"Summarize this teaching context briefly:\n\n{summary}"
        )

    # 3. FINAL LECTURE
    script = generate_script(summaries)
    
    final_script = trim_to_length(script)
    
    return final_script

# Save the script to a file
def save_script(script, path):
    with open(path, "w") as f:
        f.write(script)

# Main function

def main():
    parser = argparse.ArgumentParser(description="Convert notebook to lecture script")
    parser.add_argument("input", help="Path to input notebook (.ipynb)")
    parser.add_argument(
        "-o", "--output",
        help="Output script path",
        default=None
    )

    args = parser.parse_args()

    input_path = args.input

    if not os.path.exists(input_path):
        print(f"Error: file '{input_path}' not found")
        return

    if args.output:
        output_path = args.output
    else:
        filename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"scripts/{filename}.txt"

    script = notebook_to_script(input_path)
    save_script(script, output_path)

    print(f"Script saved to {output_path}")

if __name__ == "__main__":
    main()
