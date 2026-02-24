Pools and connections
=====================

This page explains how HTTP sessions and TCP connections are shared between a pool and its sub-pools (e.g. from :meth:`~aurora_swarm.pool.AgentPool.by_tag`, :meth:`~aurora_swarm.pool.AgentPool.sample`), and how to avoid "Unclosed connector" warnings.

Concepts
--------

**Connection**
  A single TCP link from your orchestrator process to one agent (one host:port). Many vLLM servers imply many such connections; the library reuses them via a connection pool.

**Session and connector**
  There is **one** `aiohttp.ClientSession` (and one `TCPConnector`) per **pool tree**: the pool you create with ``async with VLLMPool(...) as pool`` plus any sub-pools from ``pool.by_tag()``, ``pool.sample()``, etc. Sub-pools share that single session; they do not create their own.

**connector_limit**
  The connector is a *connection pool*: it can open up to ``connector_limit`` TCP connections (default 1024) to your many agents. So you have one logical "client" (session + connector) per pool tree, but many actual TCP connections to the many servers. ``connector_limit`` is set only in the constructor (e.g. ``VLLMPool(..., connector_limit=1024)``); there is no property getter. It is a parameter of both :class:`~aurora_swarm.pool.AgentPool` and :class:`~aurora_swarm.vllm_pool.VLLMPool`.

**Lifecycle**
  The **parent** pool creates the session when you enter the ``async with`` block. All sub-pools hold a reference to that same session. When the block exits, only the parent's :meth:`~aurora_swarm.pool.AgentPool.close` runs, which closes the shared session and connector. Always use the pool as a context manager so the session is closed and you avoid "Unclosed connector" warnings.

Diagram
-------

Relationship between parent pool, sub-pools, and the shared session:

.. code-block:: text

   User:   async with VLLMPool(endpoints, ...) as pool:
                    |
                    v
   +-----------------------------------------------------------------------------+
   |  PARENT POOL (e.g. pool from "async with")                                  |
   |  _session -----------------------------------------+                        |
   |  _semaphore (concurrency limit)                     |  __aenter__:          |
   |  _endpoints = [all agents]                          |  creates session      |
   |  _connector_limit, _timeout                         |  __aexit__: closes it |
   +----------------------------------------------------|-----------------------+
                                                        v
   +----------------------------------------------------------------------------+
   |  SINGLE aiohttp.ClientSession (shared)                                     |
   |  +-- TCPConnector(limit=connector_limit)  <- connection pool for HTTP      |
   +----------------------------------------------------------------------------+
        ^                    ^                    ^
        |                    |                    |
   child._session       child._session       child._session
   (same reference)     (same reference)     (same reference)
        |                    |                    |
   +----+--------+      +----+--------+      +----+--------+
   | SUB-POOL    |      | SUB-POOL    |      | SUB-POOL    |
   | (e.g.       |      | (e.g.       |      | (e.g.       |
   | by_tag      |      | by_tag      |      | sample)     |
   | "hypoth")   |      | "critiq")   |      |             |
   |             |      |             |      |             |
   | _session, _semaphore shared with parent                |
   | _endpoints = filtered subset only                      |
   +-------------+      +-------------+      +-------------+

Summary: one session (and one connector) per pool tree; many TCP connections (up to ``connector_limit``) to your many agents; only the parent's ``close()`` is called on context exit.
