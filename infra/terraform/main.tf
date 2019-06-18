variable "default-region" {
  default = "europe-west2"
}

variable "default-zone" {
  default = "europe-west2-a"
}

provider "google" {
  project = "myrtlespeech"
  version = "~> 2.8"
  region  = "${var.default-region}"
  zone    = "${var.default-zone}"
}

resource "google_container_cluster" "myrtlespeech" {
  name     = "myrtlespeech"
  location = "${var.default-zone}"

  remove_default_node_pool = true
  initial_node_count       = 1

  master_auth {
    username = ""
    password = ""

    client_certificate_config {
      issue_client_certificate = false
    }
  }
}

resource "google_container_node_pool" "n1-standard-1" {
  name     = "n1-standard-1"
  location = "${var.default-zone}"
  cluster  = "${google_container_cluster.myrtlespeech.name}"

  autoscaling {
    min_node_count = 0
    max_node_count = 5
  }

  initial_node_count = 2

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = "n1-standard-1"

    oauth_scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
    ]
  }
}

resource "google_container_node_pool" "n1-standard-4" {
  name     = "n1-standard-4"
  location = "${var.default-zone}"
  cluster  = "${google_container_cluster.myrtlespeech.name}"

  autoscaling {
    min_node_count = 0
    max_node_count = 5
  }

  initial_node_count = 0

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = "n1-standard-4"

    oauth_scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
    ]
  }
}

# The following outputs allow authentication and connectivity to the GKE
# Cluster by using certificate-based authentication.
output "client_certificate" {
  value = "${google_container_cluster.myrtlespeech.master_auth.0.client_certificate}"
}

output "client_key" {
  value = "${google_container_cluster.myrtlespeech.master_auth.0.client_key}"
}

output "cluster_ca_certificate" {
  value = "${google_container_cluster.myrtlespeech.master_auth.0.cluster_ca_certificate}"
}
