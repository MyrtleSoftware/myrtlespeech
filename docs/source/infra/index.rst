.. TODO::

    1. Kubernetes security (RBAC, ...)!


==============
Infrastructure
==============

Creating and maintaining a scalable, reliable CI system is hard so we'll use a
managed service to abstract this away.

This manage service is currently `GitLab CI
<https://about.gitlab.com/product/continuous-integration/>`_.

A `Google Kuberentes Engine <https://cloud.google.com/kubernetes-engine/>`_
(GKE) Kubernetes cluster is used to run the GitLab CI Jobs via GitLab's
`Kubernetes executor
<https://docs.gitlab.com/runner/executors/kubernetes.html>`_. The cluster is
configured and created from a set of declarative, version controlled scripts
using `Terraform <https://www.terraform.io/>`_. The Kubernetes executor is
installed into the cluster using `Helm <https://helm.sh>`_.


Google Cloud Platform (GCP) Project
------------------------------------

`Google Cloud <https://cloud.google.com/>`_ is the cloud provider.

   "Google Cloud Platform (GCP) projects form the basis for creating, enabling,
   and using all GCP services including managing APIs, enabling billing, adding
   and removing collaborators, and managing permissions for GCP resources."

   -- `Creating and Managing Projects <https://cloud.google.com/resource-manager/docs/creating-managing-projects>`_

A GCP project is necessary to use GCP resources. A project can be created using
the `gcloud command-line tool <https://cloud.google.com/sdk/gcloud/>`_:

.. code-block:: bash

   $ gcloud projects create myrtlespeech --set-as-default --organization 842401732689

.. note::

    :literal:`myrtlespeech` in the previous command denotes the project ID.
    GCP project IDs must be unique. Change :literal:`myrtlespeech` if the above
    command fails with an error such as:

    .. code-block:: text

        ERROR: (gcloud.projects.create) Project creation failed. The project ID you specified is already in use by another project. Please try an alternative ID.

.. note::

    :literal:`842401732689` in the previous command denotes the ID for the
    ``myrtle.ai`` organisation.


Terraform
----------

`Terraform <https://www.terraform.io>`_ manages GCP resources within the GCP
project:

   "Terraform is a tool for building, changing, and versioning infrastructure
   safely and efficiently."

   -- `Introduction to Terraform <https://www.terraform.io/intro/index.html>`_

The desired infrastructure is described in a set of files which are known as a
Terraform *configuration*. Each resource is described in Terraform's own
declarative language.

Configurations are applied using the Terraform command-line program. The
Terraform command-line tool was automatically installed when the Conda
development environment was :ref:`created <install>`.

Steps to be done before applying the configuration:

1. Create a `service account
   <https://cloud.google.com/iam/docs/understanding-service-accounts>`_ for
   Terraform.
2. Setup Terraform *remote*.

Service Account
~~~~~~~~~~~~~~~~

  "A service account is a special type of Google account that belongs to your
  application or a virtual machine (VM), instead of to an individual end user.
  Your application assumes the identity of the service account to call Google
  APIs, so that the users aren't directly involved. A service account can have
  zero or more pairs of service account keys, which are used to authenticate to
  Google."

  -- `Understanding service accounts <https://cloud.google.com/iam/docs/understanding-service-accounts>`_

Create a service account for Terraform to use:

.. code-block:: bash

   $ gcloud iam service-accounts create terraform \
         --display-name "Terraform"

Create a key to facilitate authentication:

.. code-block:: bash

   $ gcloud iam service-accounts keys create \
         --iam-account terraform@myrtlespeech.iam.gserviceaccount.com \
         ~/.keys/gcp-myrtlespeech-terraform.json

.. note::

   Change :literal:`myrtlespeech` in the :literal:`--iam-account` to match the
   project ID you're using if it is different.

Set the :literal:`GOOGLE_APPLICATION_CREDENTIALS` environment variable to the
location of the key created above. The GCP client libraries will use the
credentials at this path. See `Setting Up Authentication for Server to Server
Production Applications
<https://cloud.google.com/docs/authentication/production>`_.

.. code-block:: bash

   $ export GOOGLE_APPLICATION_CREDENTIALS=~/.keys/gcp-myrtlespeech-terraform.json

The service account for Terraform needs to have the minimal set of permissions
enabled for it to manage the GCP resources required (following the principle of
least privilege).

.. code-block:: bash

   $ roles=(compute.admin container.clusterAdmin file.editor iam.serviceAccountUser)
   $ for role in ${roles[@]}; do \
         gcloud projects add-iam-policy-binding myrtlespeech \
             --member serviceAccount:terraform@myrtlespeech.iam.gserviceaccount.com \
             --role roles/${role}; \
     done

The following APIs also need to be enabled:

.. code-block:: bash

   $ gcloud services enable \
         compute.googleapis.com \
         container.googleapis.com \
         cloudresourcemanager.googleapis.com \
         file.googleapis.com

Remote
~~~~~~~

Terraform requires storing some state about the current infrastructure and
configuration. This state can be stored in a local file but it is preferable to
store it a remote location (i.e. cloud object store).

A remote `backend <https://www.terraform.io/docs/backends/index.html>`_ will be
used to store the state. This needs to be manually created before using
Terraform to automatically create the rest of the infrastructure (the backend
configuration cannot contain interpolations).

A `Google Cloud Storage <https://cloud.google.com/storage/>`_ bucket is used as
a remote backend to handle the storage and locking of the state. This can be
achieved using the `gsutil <https://cloud.google.com/storage/docs/gsutil>`_
command-line program:

.. code-block:: bash

   $ gsutil mb -c multi_regional -l eu gs://myrtlespeech-terraform-state

Enable versioning `to allow for state recovery in the case of accidental
deletions and human error.
<https://www.terraform.io/docs/backends/types/gcs.html>`_:

.. code-block:: bash

   $ gsutil versioning set on gs://myrtlespeech-terraform-state

Grant the objectAdmin role to the service account for this bucket:

.. code-block:: bash

   $ gsutil iam ch serviceAccount:terraform@myrtlespeech.iam.gserviceaccount.com:objectAdmin gs://myrtlespeech-terraform-state

The roles/storage.objectAdmin role:

   "Grants full control over objects, including listing, creating, viewing, and
   deleting objects."

   -- `Cloud IAM Roles for Cloud Storage <https://cloud.google.com/storage/docs/access-control/iam-roles>`_

Initialize the remote:

.. code-block:: bash

   $ cd infra/terraform  # must be in this dir so terraform reads files
   $ terraform init

Kubernetes Engine
~~~~~~~~~~~~~~~~~~

A GKE cluster can now be created by applying the terraform configuration in the
:literal:`infra/terraform` directory:

.. code-block:: bash

    $ pwd
    /home/user/myrtlespeech/infra/terraform
    $ terraform apply  # check and accept (or decline) execution plan

.. note::

    If ``terraform`` raises an error about permissions that you've already
    given the service account then you may need to remove and readd the the
    role bindings.

    This appears to occur when a service account with the same name and set of
    bindings has previously existed and has since been deleted.

Configure :literal:`kubectl` to point to this cluster:

.. code-block:: bash

    $ gcloud container clusters get-credentials myrtlespeech --region europe-west2-a


Helm
-----

Helm is a package manager for Kubernetes.

    "Helm helps you manage Kubernetes applications --- Helm Charts help you
    define, install, and upgrade even the most complex Kubernetes application."

    -- `What is Helm? <https://helm.sh/>`_

Installation
~~~~~~~~~~~~~

    "There are two parts to Helm: The Helm client (:literal:`helm`) and the
    Helm server (Tiller)."

    -- `Installing Helm <https://helm.sh/docs/using_helm/#installing-helm>`_

Both parts need to be installed.

The Helm client is a binary that is run on the local device. It is included by
default in the :literal:`myrtlespeech` Conda environment.

Tiller, which runs inside of the Kubernetes cluster and manages releases
(installations) of charts, can be installed into the cluster using the Helm
client. First create a service account for Tiller:

.. code-block:: bash

    $ kubectl create serviceaccount --namespace kube-system tiller
    $ kubectl create clusterrolebinding tiller-cluster-rule --clusterrole=cluster-admin --serviceaccount=kube-system:tiller
    $ kubectl patch deploy --namespace kube-system tiller-deploy -p '{"spec":{"template":{"spec":{"serviceAccount":"tiller"}}}}'

and then install it:

.. code-block:: bash

    $ helm init --history-max 200

More information can be found `here
<https://helm.sh/docs/using_helm/#initialize-helm-and-install-tiller>`_.

Charts
~~~~~~~

    "Helm uses a packaging format called *charts*. A chart is a collection of
    files that describe a related set of Kubernetes resources. A single chart
    might be used to deploy something simple, like a memcached pod, or
    something complex, like a full web app stack with HTTP servers, databases,
    caches, and so on."

    "A chart is organized as a collection of files inside of a directory. The
    directory name is the name of the chart (without versioning information)."

    -- `Charts <https://helm.sh/docs/developing_charts/#charts>`_

The Repaper chart configures the necessary software to run on the GKE cluster.

See the :literal:`infra/myrtlespeech/` directory for more information about the
chart.

Add the registration token to ``values.yaml`` and then use the the Helm client
library to install the chart ensure to change to the repaper chart directory:

.. code-block:: bash

    $ pwd
    /home/user/.../repaper/infra/myrtlespeech
    $ helm repo add gitlab https://charts.gitlab.io
    $ helm dep update
    $ helm install . --namespace gitlab-ci


GitLab CI
~~~~~~~~~

Helm installed a `GitLab Runner instance
<https://docs.gitlab.com/runner/install/kubernetes.html>`_ into the cluster
that will provision a new pod for each GitLab CI/CD job it receives.

See ``.gitlab-ci.yml`` for the full configuration.

A schedule should be manually configured in the GitLab CI web UI to build the
master branch each night at 0200 to catch any regression.
