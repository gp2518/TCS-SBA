#Git Answers Submission
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def basic_lcel_chain(topic: str) -> str:
    """Returns a one-paragraph explanation of the given topic."""

    prompt = ChatPromptTemplate.from_template(
        "Explain the topic '{topic}' in one clear paragraph."
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
    )

    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain.invoke({"topic": topic})

result = basic_lcel_chain("quantum computing")
print(result)