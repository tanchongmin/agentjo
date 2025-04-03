################################
## User Contribution: Richard ##
################################

from agentjo.memory import MemoryTemplate

class SqliteMemory(MemoryTemplate):
    ''' Memory in a SQL database, retrieves using OpenAI embeddings 
    Contributed by Richard. See https://github.com/tanchongmin/agentjo/blob/main/contrib/Memory/SqliteMemory.ipynb for usage details'''
    def __init__(self, memory_list = None, top_k: int = 3, sqlite_database=":memory:", embedding_procedure = None):
        ''' Inputs:
        memory_list: Initial list of memories to store
        top_k: Number of memories to retrieve
        sqlite_database: path to the sqlite database
        embedding_procedure: the procedure to get embedding value for text
        '''
        import sqlite3
        import sqlite_vec
        
        self.top_k = top_k
        self.embedding_procedure = embedding_procedure
        self.embedding_length = len(self.embedding_procedure("foo"))

        self.database = sqlite3.connect(sqlite_database)
        self.database.enable_load_extension(True)
        sqlite_vec.load(self.database)
        self.database.enable_load_extension(False)
        cur = self.database.cursor()
        cur.execute(f"CREATE VIRTUAL TABLE vec_memories USING vec0(memory_embedding float[{self.embedding_length}])")
        cur.execute("create table memories(memory text)")
        if memory_list is not None:
            self.append(memory_list)

    def append(self, memory_list, mapper=None):
        """Adds a list of memories"""
        if not isinstance(memory_list, list):
            memory_list = [memory_list]
        cur = self.database.cursor()
        for memory in memory_list:
          embedding = self.embedding_procedure(memory)
          cur.execute("INSERT INTO memories(memory) VALUES (?)", [memory])
          cur.execute('insert into vec_memories(rowid, memory_embedding) VALUES (?, ?)', [cur.lastrowid, sqlite_vec.serialize_float32(embedding)])

    def remove(self, memory_to_remove):
        """Removes a memory"""
        cur = self.database.cursor()
        result = cur.execute("select rowid from memories WHERE memory = ?;", [memory_to_remove])
        rowid = result.fetchone()[0]
        cur.execute("delete from memories WHERE rowid = ?;", [rowid])
        cur.execute("delete from vec_memories WHERE rowid = ?;", [rowid])

    def reset(self):
        """Clears all memory"""
        cur = self.database.cursor()
        cur.execute("DELETE FROM vec_memories;")
        cur.execute("DELETE FROM memories;")
        
    def retrieve(self, task: str) -> list:
        task_embedding = self.embedding_procedure(task)
        """Performs retrieval of top_k similar memories according to embedding similarity"""
        cur = self.database.cursor()
        rows = cur.execute(
            """
              with matches as (SELECT rowid, distance FROM vec_memories WHERE memory_embedding MATCH ? AND k = ? ORDER BY distance)
              select memory from matches left join memories on memories.rowid = matches.rowid;
            """,
            [sqlite_vec.serialize_float32(task_embedding), self.top_k]
        ).fetchall()
        return [row[0] for row in rows]

class AsyncSqliteMemory(MemoryTemplate):
    ''' Memory in a SQL database, retrieves using OpenAI embeddings '''
    @classmethod
    async def create(cls, memory_list = None, top_k: int = 3, sqlite_database=":memory:", embedding_procedure = None):
        self = cls()
        ''' Inputs:
        memory_list: Initial list of memories to store
        top_k: Number of memories to retrieve
        sqlite_database: path to the sqlite database
        embedding_procedure: the procedure to get embedding value for text
        '''
        import sqlite3
        import sqlite_vec
        import asyncio
        self.top_k = top_k
        self.embedding_procedure = embedding_procedure
        self.embedding_length = len(await embedding_procedure("foo"))

        self.database = sqlite3.connect(sqlite_database)
        self.database.enable_load_extension(True)
        sqlite_vec.load(self.database)
        self.database.enable_load_extension(False)
        cur = self.database.cursor()
        cur.execute(f"CREATE VIRTUAL TABLE vec_memories USING vec0(memory_embedding float[{self.embedding_length}])")
        cur.execute("create table memories(memory text)")
        if memory_list is not None:
            await self.append(memory_list)
        return self

    async def append(self, memory_list, mapper=None):
        """Adds a list of memories"""
        if not isinstance(memory_list, list):
            memory_list = [memory_list]
        cur = self.database.cursor()
        for memory in memory_list:
          embedding = await self.embedding_procedure(memory)
          cur.execute("INSERT INTO memories(memory) VALUES (?)", [memory])
          cur.execute('insert into vec_memories(rowid, memory_embedding) VALUES (?, ?)', [cur.lastrowid, sqlite_vec.serialize_float32(embedding)])

    def remove(self, memory_to_remove):
        """Removes a memory"""
        cur = self.database.cursor()
        result = cur.execute("select rowid from memories WHERE memory = ?;", [memory_to_remove])
        rowid = result.fetchone()[0]
        cur.execute("delete from memories WHERE rowid = ?;", [rowid])
        cur.execute("delete from vec_memories WHERE rowid = ?;", [rowid])

    def reset(self):
        """Clears all memory"""
        cur = self.database.cursor()
        cur.execute("DELETE FROM vec_memories;")
        cur.execute("DELETE FROM memories;")
        
    async def retrieve(self, task: str) -> list:
        task_embedding = await self.embedding_procedure(task)
        """Performs retrieval of top_k similar memories according to embedding similarity"""
        cur = self.database.cursor()
        rows = cur.execute(
            """
              with matches as (SELECT rowid, distance FROM vec_memories WHERE memory_embedding MATCH ? AND k = ? ORDER BY distance)
              select memory from matches left join memories on memories.rowid = matches.rowid;
            """,
            [sqlite_vec.serialize_float32(task_embedding), self.top_k]
        ).fetchall()
        return [row[0] for row in rows]