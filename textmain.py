# main.py

from mia import Mia  # Assuming 'mia.py' contains the main class and functionality for Mia

def main():
    # Initialize Mia
    mia = Mia()
    mia.speak("Hello! I'm Mia, your personal assistant. How can I help you today?")

    # Main interaction loop
    while True:
        user_input = input("\nYou: ")

        # Simple keyword-based routing
        if "neuroscience" in user_input.lower():
            mia.answer_neuroscience_question(user_input)

        elif "math" in user_input.lower():
            mia.answer_math_question(user_input)

        elif "astronomy" in user_input.lower():
            mia.answer_astronomy_question(user_input)

        elif "seismology" in user_input.lower():
            mia.answer_seismology_question(user_input)

        elif "medicine" in user_input.lower():
            mia.answer_medicine_question(user_input)

        elif "analyze image" in user_input.lower():
            image_path = input("Please provide the path of the image: ")
            analysis = mia.analyze_image(image_path)
            print(f"Image Analysis Result: {analysis}")

        elif "external data" in user_input.lower() or "fetch data" in user_input.lower():
            query = input("What data are you looking for? ")
            data = mia.fetch_external_data(query)
            print(f"Fetched Data: {data}")

        elif "sensitive info" in user_input.lower():
            info = input("Enter the sensitive information: ")
            mia.handle_sensitive_info(info)

        elif "backup" in user_input.lower():
            mia.perform_backup()

        elif "problem" in user_input.lower():
            mia.solve_problem(user_input)

        elif "analyze" in user_input.lower():
            data = input("Enter the data for analysis: ")
            analysis_result = mia.analyze_data(data)
            print(f"Data Analysis Result: {analysis_result}")

        elif "learn from interaction" in user_input.lower() or "feedback" in user_input.lower():
            username = input("Enter your name: ")
            feedback = input("Provide feedback for Mia: ")
            mia.learn_from_interaction(username, feedback)

        elif "gratitude" in user_input.lower() or "thank you" in user_input.lower():
            mia.show_gratitude()

        elif "affection" in user_input.lower():
            mia.express_affection()

        elif "suggest activity" in user_input.lower() or "suggest something" in user_input.lower():
            mia.provide_activity_suggestions()

        elif "exit" in user_input.lower() or "bye" in user_input.lower():
            mia.speak("Goodbye! It was a pleasure assisting you.")
            break

        else:
            # Default action for general conversation
            response = mia.generate_response(user_input)
            print(f"Mia: {response}")

if __name__ == "__main__":
    main()