pipeline {
    agent any

    stages {

        stage('Checkout Code') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/https://2024aa05593-trips/mlops-heart-disease.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                python --version
                pip install -r requirements.txt
                '''
            }
        }

        stage('Run Unit Tests') {
            steps {
                sh 'pytest tests/'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t heart-api:latest .'
            }
        }

        stage('Run Docker Container') {
            steps {
                sh '''
                docker stop heart-api || true
                docker rm heart-api || true
                docker run -d -p 8000:8000 --name heart-api heart-api:latest
                '''
            }
        }
    }
}
