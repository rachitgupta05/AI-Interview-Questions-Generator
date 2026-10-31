
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Initialize Gemini client
client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(timeout=30000)
)
MODEL_NAME = "gemini-3.5-flash-lite"

def generate_questions(subject, difficulty, num_questions=10, random_seed=None):
    """Generate interview questions using Gemini."""

    prompt = f"""
    Generate exactly {num_questions} unique technical interview
    questions about {subject}.

    Difficulty: {difficulty}

    Requirements:
    - Questions should be relevant to technical interviews.
    - Avoid duplicate questions.
    - Match the requested difficulty.
    - Return only a valid JSON array of strings.

    Example:
    ["What is Java?", "Explain inheritance in Java."]
    """
    print("Sending request to Gemini...")
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.7
        )
    )
    print("Gemini response received.")

    questions = json.loads(response.text)

    if not isinstance(questions, list):
        raise ValueError("Gemini returned an invalid question format")

    return [str(q).strip() for q in questions if str(q).strip()][:num_questions]


def generate_model_answer(question, subject, difficulty):
    """Generate a reference answer using Gemini."""

    prompt = f"""
    You are an experienced technical interviewer.

    Subject: {subject}
    Difficulty: {difficulty}
    Question: {question}

    Generate a technically accurate, clear and detailed
    reference answer suitable for an interview candidate.
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4
        )
    )

    return response.text.strip()


def generate_model_answers_batch(questions, subject, difficulty):
    """Generate reference answers for all questions in one Gemini request."""

    items = [
        {"id": i, "question": question}
        for i, question in enumerate(questions)
    ]

    prompt = f"""
    You are an experienced technical interviewer.

    Subject: {subject}
    Difficulty: {difficulty}

    Generate a technically accurate, clear and detailed
    reference answer for every interview question.

    Return ONLY a valid JSON array in this format:
    [
        {{
            "id": 0,
            "answer": "Detailed reference answer"
        }}
    ]

    Questions:
    {json.dumps(items, ensure_ascii=False)}

    IMPORTANT:
    - Generate an answer for every question.
    - Preserve the original question IDs.
    - Never return an empty answer.
    - Keep answers appropriate for the specified difficulty.
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3
        )
    )

    results = json.loads(response.text)

    if not isinstance(results, list) or len(results) != len(questions):
        raise ValueError("Gemini returned an incomplete set of model answers.")

    answers = {}

    for result in results:
        index = int(result["id"])
        answer = str(result.get("answer", "")).strip()

        if index < 0 or index >= len(questions) or index in answers:
            raise ValueError("Gemini returned an invalid question ID.")

        if not answer:
            raise ValueError(f"Model answer missing for question {index + 1}.")

        answers[index] = answer

    if len(answers) != len(questions):
        raise ValueError("Gemini did not generate every model answer.")

    return [answers[i] for i in range(len(questions))]


def evaluate_answer(question, user_answer, model_answer=None,
                    subject=None, difficulty=None):
    """Evaluate a candidate's answer using Gemini."""

    if not user_answer or not user_answer.strip():
        return {
            "correctness": "Incorrect",
            "feedback": "No answer provided.",
            "rating": 0
        }

    prompt = f"""
    You are a professional technical interview evaluator.

    Subject: {subject}
    Difficulty: {difficulty}

    Interview Question:
    {question}

    Expected Answer:
    {model_answer or "Generate the expected answer yourself."}

    Candidate's Answer:
    {user_answer}

    Evaluate the candidate based on:
    1. Technical accuracy
    2. Understanding of concepts
    3. Completeness
    4. Relevance

    Give a rating between 0 and 5.

    Return ONLY valid JSON in this format:
    {{
        "correctness": "Excellent",
        "feedback": "Your detailed feedback here",
        "rating": 4
    }}
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2
        )
    )

    result = json.loads(response.text)

    rating = max(0, min(5, float(result["rating"])))

    return {
        "correctness": str(result["correctness"]),
        "feedback": str(result["feedback"]),
        "rating": rating
    }


def evaluate_answers_batch(questions, answers, subject, difficulty):
    """Evaluate all interview answers using one Gemini API request."""

    items = []

    for i, (question, answer) in enumerate(zip(questions, answers)):
        items.append({
            "id": i,
            "question": question,
            "answer": answer
        })

    prompt = f"""
    You are a professional technical interview evaluator.

    Subject: {subject}
    Difficulty: {difficulty}

    Give each answer:
    - correctness: Excellent, Good, Average, or Needs Improvement
    - feedback: useful improvement feedback
    - rating: a number between 0 and 5

    IMPORTANT:
    - Feedback is mandatory for every question.
    - Never return an empty feedback string.
    - Explain what the candidate answered correctly.
    - Identify missing or incorrect concepts.
    - Suggest how the answer can be improved.
    - Provide at least 2 meaningful sentences of feedback.

    Questions and answers:
    {json.dumps(items, ensure_ascii=False)}

    Return ONLY a JSON array in this format:
    [
        {{
            "id": 0,
            "correctness": "Good",
            "feedback": "Your feedback",
            "rating": 4
        }}
    ]

    Evaluate every item and preserve its id.
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2
        )
    )

    results = json.loads(response.text)

    print("\n========== GEMINI DEBUG ==========")
    print("Expected questions:", len(questions))
    print("Returned evaluations:", len(results) if isinstance(results, list) else "Invalid format")
    print("=================================\n")

    if not isinstance(results, list) or len(results) != len(questions):
        raise ValueError("Gemini returned an invalid evaluation result.")

    evaluations = {}

    for result in results:
        index = int(result["id"])

        if index < 0 or index >= len(questions) or index in evaluations:
            raise ValueError("Gemini returned an invalid question ID.")

        rating = max(0, min(5, float(result["rating"])))

        correctness = str(
            result.get("correctness", "Needs Improvement")
        ).strip()

        feedback = str(result.get("feedback", "")).strip()

        if not feedback:
            feedback = (
                "Feedback was not provided by the AI evaluator. "
                "Please review your answer and try again."
            )

        # Store each evaluation inside the loop
        evaluations[index] = {
            "correctness": correctness,
            "feedback": feedback,
            "rating": rating
        }

    if len(evaluations) != len(questions):
        raise ValueError("Gemini did not evaluate every question.")

    return [evaluations[i] for i in range(len(questions))]
