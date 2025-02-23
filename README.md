![1731997275507-SBI Banner_2](https://github.com/user-attachments/assets/83029fbb-44e7-47f1-a68e-60b42a70087a)
# SBI Life Insurance AI Personalization Platform

Enhance the customer experience with our AI-driven, real-time personalization platform designed for SBI Life Insurance. Our solution leverages Machine Learning, Knowledge Graphs, and Reinforcement Learning with Human Feedback (RLHF) to deliver dynamic policy recommendations, adjust premiums based on risk, and continuously adapt to evolving customer needs.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training & RLHF](#model-training--rlhf)
- [Tagging System](#tagging-system)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

Traditional recommendation systems rely on static, persona-based models that quickly become outdated. Our platform addresses this limitation by using:

- **Knowledge Graphs:** Organize complex relationships between customers, policies, and behaviors.
- **Real-time Dynamic Tagging:** Continuously update user profiles based on interactions, financial activity, and preferences.
- **RLHF (Reinforcement Learning with Human Feedback):** Fine-tune recommendations through user feedback, ensuring that policy suggestions and premium adjustments stay relevant.

The result is an intelligent system that not only provides smarter, faster, and more personalized insurance offerings but also enhances customer satisfaction, retention, and loyalty.

## Features

- **Dynamic Customer Profiling:** Leverage a robust Knowledge Graph to capture and update relationships between customers and their insurance needs.
- **Personalized Recommendations:** Use RLHF to adapt policy recommendations based on real-time feedback.
- **Premium Adjustment:** Dynamically adjust premiums according to risk factors and predictive analytics.
- **Comprehensive Tagging System:** Automatically tag and categorize user interactions to inform future recommendations.
- **Integrated Web Application:** A Django-based application that provides a user-friendly interface for both customers and administrators.
- **Model Training & Evaluation:** Jupyter Notebook and RLHF scripts for training and fine-tuning machine learning models.

## File Structure

Below is an overview of the project structure:

```
├── Hello
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── settings.cpython-310.pyc
│   │   ├── urls.cpython-310.pyc
│   │   └── wsgi.cpython-310.pyc
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── db.sqlite3
├── home
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── urls.cpython-310.pyc
│   │   ├── utils.cpython-310.pyc
│   │   └── views.cpython-310.pyc
│   ├── admin.py
│   ├── apps.py
│   ├── migrations
│   │   └── __init__.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── manage.py
├── static
│   ├── Back.jpg
│   ├── SBI.png
│   └── test.txt
└── templates
    ├── chatbot.html
    ├── dump.html
    ├── image.png
    ├── new.html
    └── real_chatbot.html
rlhf
├── Actornetwork.py
├── chatbotenv.py
├── criticnetwork.py
├── main.py
└── ppoAgent.py
user-tags
└── tagging-system.py
venv
.env
.gitignore
Model Training Code.ipynb
requirements.txt
user_tags.json
```

**Directory Breakdown:**

- **Hello/**  
  Contains the Django project configuration files including settings, URL routing, and WSGI/ASGI configuration.

- **home/**  
  A Django application that holds the core business logic (views, models, admin configurations, and URL routes) for the platform.

- **static/** and **templates/**  
  Store the static assets (images, text files) and HTML templates respectively for rendering the front-end components.

- **rlhf/**  
  Contains the RLHF implementation scripts, including actor-critic networks and PPO agent, used for fine-tuning recommendation policies.

- **user-tags/**  
  Houses the dynamic tagging system script that manages real-time updates to user profiles.

- **Other files:**  
  - `manage.py`: Django management script.
  - `db.sqlite3`: The default SQLite database.
  - `Model Training Code.ipynb`: Jupyter Notebook for training models.
  - `requirements.txt`: Lists Python dependencies.
  - `.env` and `.gitignore`: Environment variables and files/directories to exclude from version control.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Set Up Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   Create a `.env` file in the root directory and add your necessary environment configurations (e.g., secret keys, database settings).

5. **Apply Migrations:**

   ```bash
   python manage.py migrate
   ```

## Usage

### Running the Web Application

To start the Django development server, run:

```bash
python manage.py runserver
```

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser to view the application.

### Interacting with the System

- **Customer Interactions:**  
  Use the provided HTML templates (found in the `templates/` directory) for testing the chatbot and dynamic recommendation interfaces.

- **Admin Interface:**  
  Access the Django admin at [http://127.0.0.1:8000/admin](http://127.0.0.1:8000/admin) to manage models, view user profiles, and monitor system performance.

## Model Training & RLHF

The `rlhf/` directory includes the scripts necessary for training and fine-tuning our recommendation models using RLHF techniques:

- **Actornetwork.py & criticnetwork.py:**  
  Define the neural network architectures for the actor and critic.

- **ppoAgent.py:**  
  Contains the Proximal Policy Optimization (PPO) implementation.

- **chatbotenv.py:**  
  Sets up the environment for the RLHF framework.

- **main.py:**  
  The entry point for initiating training sessions.

Additionally, the `Model Training Code.ipynb` notebook offers an interactive environment for experimenting with model training and evaluation.

## Tagging System

The dynamic tagging system located in the `user-tags/` directory (`tagging-system.py`) continuously updates user profiles based on interactions, enabling our recommendation engine to stay current with user preferences and behaviors.

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request detailing your changes.

Please ensure your code follows the established project style and includes appropriate tests and documentation.

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.

## Contact

For further questions or suggestions, please contact [grovervishrut@gmail.com](mailto:grovervishrut@gmail.com).
