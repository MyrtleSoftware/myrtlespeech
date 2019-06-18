terraform {
  backend "gcs" {
    bucket = "myrtlespeech-terraform-state"
    prefix = "terraform/state"
  }
}
