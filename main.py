from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, predict  # Your custom pipeline

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    user_input = ""

    if request.method == 'POST':
        try:
            user_input = request.form.get('data', '').strip()

            if not user_input:
                result = "Please enter some text to predict."
            else:
                # Prepare input and make prediction
                data = CustomData(user_input)
                df = data.get_data_as_data_frame()
                prediction = predict(df)

                result = 'Hate' if prediction[0] >= 0.5 else "No Hate"
        except Exception as e:
            result = f"An error occurred: {e}"

    return render_template('index.html', result=result, user_input=user_input)


if __name__ == '__main__':
    app.run(debug=True, port=5050)

