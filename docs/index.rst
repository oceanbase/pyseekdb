.. pyseekdb documentation master file

Welcome to pyseekdb's documentation!
====================================

pyseekdb is a unified Python client for seekdb that supports embedded, server, and OceanBase modes.
It provides simple and easy-to-use APIs for vector database operations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index

Installation
============

Install pyseekdb using pip:

.. code-block:: bash

   pip install -U pyseekdb

Quick Start
===========

Embedded Mode
-------------

Connect to a local embedded seekdb instance:

.. code-block:: python

   import pyseekdb

   client = pyseekdb.Client(path="./seekdb.db", database="test")
   collection = client.get_or_create_collection("my_collection")

Remote Server Mode
------------------

Connect to a remote seekdb or OceanBase server:

.. code-block:: python

   import pyseekdb

   client = pyseekdb.Client(
       host='localhost',
       port=2881,
       tenant="sys",
       database="test",
       user="root",
       password="pass"
   )
   collection = client.get_or_create_collection("my_collection")

Admin Client
------------

Manage databases using the AdminClient:

.. code-block:: python

   import pyseekdb

   admin = pyseekdb.AdminClient(path="./seekdb.db")
   admin.create_database("new_db")
   databases = admin.list_databases()

Features
========

* **Unified API**: Single interface for embedded and remote server modes
* **Vector Operations**: Efficient vector similarity search
* **Hybrid Search**: Combine vector and full-text search
* **Embedding Functions**: Built-in support for various embedding models
* **Collection Management**: Easy collection (table) creation and management
* **Database Management**: Admin operations for database management

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
