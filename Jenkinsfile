pipeline {
  environment {
    registry = "registry.ifora.hse.ru/ml_w2v_evaluate"
    registryCredential = '2347b9bd-46aa-42ad-b775-758e40e83d80'
    dockerImage = ''
    GIT_TAG = "${BRANCH_NAME}"
  }
  agent any

  stages {

    stage('Building image') {
      steps{
        script {
          dockerImage = docker.build registry + ":$GIT_TAG"
        }
      }
    }
    stage('Push Image') {
      steps{
        script {
          docker.withRegistry('https://registry.ifora.hse.ru', registryCredential ) {
            dockerImage.push()
          }
        }
      }
    }
}
}

