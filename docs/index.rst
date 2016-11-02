Welcome to bb_binary's documentation!
========================================

.. toctree::
   :maxdepth: 2

   api/common
   api/converting
   api/parsing
   api/repository
   api/schema

We switched to `Cap'n Proto <https://capnproto.org/>`_ as data interchange format to tackle three
of our main problems:

1. IO-Operations have been a huge bottleneck in our Pipeline
2. The format has an inherent data Schema
3. There are implementations for all the major programming languages

As most of our applications are written in Python this is the recommended interface. It is also
**possible** to use *bb_binary* from other languages like C, C++ and Java, but they are not
supported right now.

The Python Module is divided into the three submoduls :doc:`api/converting`, :doc:`api/parsing` and
the :doc:`api/repository` Class.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

