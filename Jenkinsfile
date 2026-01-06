pipeline {
    agent any

    stages {

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t heart-api .'
            }
        }

        stage('Run Docker Container') {
            steps {
                sh '''
                docker rm -f heart-api || true
                docker run -d -p 8000:8000 --name heart-api heart-api
                '''
            }
        }
    }
}
