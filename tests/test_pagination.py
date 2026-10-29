import unittest
from unittest.mock import patch
from fastapi import HTTPException
from backend.app import sync_mail
import json
import os
import tempfile
from backend.app import init_db

class Response:
    def __init__(self, obj): self.raw=json.dumps(obj).encode()
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def read(self): return self.raw

class PaginationTests(unittest.TestCase):
    def test_second_page_processed(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        with patch.dict(os.environ, {"TRIAGELAB_DB":os.path.join(temp.name,"test.sqlite")}):
            init_db()
            self.check_pages()
    def check_pages(self):
        pages={'http://127.0.0.1:8025/api/v1/messages?limit=100&start=0':{'messages':[{'ID':str(i)} for i in range(100)],'total':101}, 'http://127.0.0.1:8025/api/v1/messages?limit=100&start=100':{'messages':[{'ID':'last'}],'total':101}}
        def opener(url,timeout):
            if url in pages: return Response(pages[url])
            return Response({'ID':url.rsplit('/',1)[-1],'Text':'help'})
        with patch('backend.app.urlopen',side_effect=opener), patch('backend.app.import_message',return_value=True):
            self.assertEqual(sync_mail(),101)
if __name__ == '__main__': unittest.main()
