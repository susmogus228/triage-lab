import os, sqlite3, smtplib, time, json, threading
from pathlib import Path
from contextlib import contextmanager
from email.message import EmailMessage
from email.headerregistry import Address
from html.parser import HTMLParser
from typing import Literal
from urllib.request import urlopen
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title='TriageLab API', version='0.1.0')
EVALUATION_REPORT = ROOT / 'artifacts' / 'stress_report.json'
IT_EXPERIMENT_REPORT = ROOT / 'artifacts' / 'it_experiment_report.json'
IT_V2_REPORT = ROOT / 'artifacts' / 'it_v2_report.json'
EVOLUTION_REPORT = ROOT / 'artifacts' / 'evolution_report.json'
EVOLUTION_MANIFEST = ROOT / 'artifacts' / 'evolution_data_manifest.json'
class ModelUnavailable(Exception): pass

def predict(text):
    try:
        from ml.inference import predict as model_predict
        return model_predict(text)
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        raise ModelUnavailable(str(exc)) from exc

@contextmanager
def db():
    path = Path(os.environ.get('TRIAGELAB_DB', ROOT / 'work' / 'triage.sqlite'))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10); conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()

def init_db():
    with db() as c:
        c.executescript('''CREATE TABLE IF NOT EXISTS tickets (id INTEGER PRIMARY KEY, subject TEXT NOT NULL, body TEXT NOT NULL, source TEXT NOT NULL, email TEXT, created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')), status TEXT NOT NULL, predicted_topic TEXT, confidence REAL, needs_review INTEGER NOT NULL, corrected_topic TEXT, model_version TEXT, mail_id TEXT UNIQUE);
        CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ticket_id INTEGER NOT NULL, event_type TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')));''')

def event(c, ticket_id, name, detail):
    c.execute('INSERT INTO events(ticket_id,event_type,detail) VALUES (?,?,?)',(ticket_id,name,detail))

def create_ticket(subject, body, source, email=None, mail_id=None):
    try: result = predict(subject + '\n' + body)
    except ModelUnavailable: result = {'label':None,'confidence':None,'needs_review':True,'model_version':None}
    # Conservative local review rule; raw model scores are not calibrated.
    if result.get('confidence') is not None and result['confidence'] < 0.6:
        result = dict(result, needs_review=True)
    with db() as c:
        if mail_id and c.execute('SELECT id FROM tickets WHERE mail_id=?',(mail_id,)).fetchone(): return None
        cursor=c.execute('INSERT INTO tickets(subject,body,source,email,status,predicted_topic,confidence,needs_review,model_version,mail_id) VALUES (?,?,?,?,?,?,?,?,?,?)', (subject,body,source,email,'needs_review' if result['needs_review'] else 'classified',result['label'],result['confidence'],int(result['needs_review']),result.get('model_version'),mail_id))
        tid=cursor.lastrowid
        event(c,tid,'received',f'Received from {source}')
        event(c,tid,'prediction',f"Model: {result['label']}" if result['label'] else 'Model unavailable; manual review required')
        return dict(c.execute('SELECT * FROM tickets WHERE id=?',(tid,)).fetchone())

class TicketIn(BaseModel):
    subject: str = Field(min_length=1,max_length=300)
    body: str = Field(min_length=1,max_length=10000)
    source: Literal['storefront','help-center','email']
    email: str | None = Field(default=None,max_length=254)
class Correction(BaseModel): corrected_topic: str = Field(min_length=1,max_length=120)
class TestEmail(BaseModel):
    subject: str = Field(min_length=1,max_length=300)
    body: str = Field(min_length=1,max_length=10000)
    email: str = Field(default='visitor@example.com',max_length=254)

    @field_validator('subject', 'email')
    @classmethod
    def validate_email_header(cls, value):
        if '\r' in value or '\n' in value:
            raise ValueError('Email header must not contain line breaks')
        return value

    @field_validator('email')
    @classmethod
    def validate_sender(cls, value):
        try:
            address = Address(addr_spec=value)
            if not address.username or not address.domain or address.addr_spec != value:
                raise ValueError('Use a single valid test email address')
        except (TypeError, ValueError):
            raise ValueError('Use a single valid test email address') from None
        return value

def poll_mail():
    while True:
        try:
            sync_mail()
        except (HTTPException, sqlite3.Error):
            pass
        time.sleep(3)

@app.on_event('startup')
def startup():
    init_db()
    with db() as c:
        c.execute("UPDATE tickets SET needs_review=1,status='needs_review' WHERE status='classified' AND confidence < 0.6")
    threading.Thread(target=poll_mail, daemon=True, name='mailpit-poller').start()
@app.get('/api/health')
def health(): return {'status':'ok'}
@app.get('/api/tickets')
def tickets():
    with db() as c: return {'tickets':[dict(row) for row in c.execute('SELECT * FROM tickets ORDER BY id DESC')]}
@app.post('/api/tickets',status_code=201)
def post_ticket(item:TicketIn):
    if not item.subject.strip() or not item.body.strip(): raise HTTPException(422,'Subject and message are required')
    return {'ticket':create_ticket(item.subject.strip(),item.body.strip(),item.source,item.email)}
@app.get('/api/tickets/{ticket_id}')
def get_ticket(ticket_id:int):
    with db() as c:
        ticket=c.execute('SELECT * FROM tickets WHERE id=?',(ticket_id,)).fetchone()
        if not ticket: raise HTTPException(404,'Ticket not found')
        events=[dict(x) for x in c.execute('SELECT event_type,detail,created_at FROM events WHERE ticket_id=? ORDER BY id',(ticket_id,))]
        return {'ticket':dict(ticket),'events':events}
@app.patch('/api/tickets/{ticket_id}')
def correct(ticket_id:int,item:Correction):
    with db() as c:
        if not c.execute('SELECT id FROM tickets WHERE id=?',(ticket_id,)).fetchone(): raise HTTPException(404,'Ticket not found')
        c.execute('UPDATE tickets SET corrected_topic=?, needs_review=0,status=? WHERE id=?',(item.corrected_topic,'reviewed',ticket_id))
        event(c,ticket_id,'correction',f'Operator selected: {item.corrected_topic}')
    return get_ticket(ticket_id)
@app.get('/api/model')
def model():
    try:
        from ml.inference import get_report
        from ml.inference import ARTIFACT
        if not ARTIFACT.is_file(): raise RuntimeError('Model artifact is unavailable')
        report = get_report()
        test = report.get('test', {})
        return {'available':True, 'model_name':report.get('model_version'), 'dataset':report.get('dataset', {}).get('name', 'Unknown'), 'labels':test.get('class_order', []), 'metrics':{'accuracy':test.get('accuracy'),'macro_f1':test.get('macro_f1'),'confusion_matrix':test.get('confusion_matrix')}, 'limitations':report.get('notes', []), 'report':report}
    except (ImportError,FileNotFoundError,RuntimeError):
        return {'available':False,'limitations':['The model artifact has not been trained yet.']}

@app.get('/api/evaluation')
def evaluation():
    if not EVALUATION_REPORT.is_file():
        return {'available':False, 'reason':'No independent evaluation report has been generated.'}
    try:
        return json.loads(EVALUATION_REPORT.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise HTTPException(503,'Evaluation report is unreadable')

@app.get('/api/evolution')
def evolution():
    if not EVOLUTION_REPORT.is_file():
        return {'available':False, 'reason':'No evolution report has been generated.'}
    try:
        report = json.loads(EVOLUTION_REPORT.read_text(encoding='utf-8'))
        report['data_manifest'] = json.loads(EVOLUTION_MANIFEST.read_text(encoding='utf-8')) if EVOLUTION_MANIFEST.is_file() else None
        return report
    except (OSError, ValueError):
        raise HTTPException(503,'Evolution report is unreadable')

@app.get('/api/it-experiment')
def it_experiment():
    if not IT_EXPERIMENT_REPORT.is_file():
        return {'available':False, 'reason':'No IT ticket experiment report has been generated.'}
    try:
        return json.loads(IT_EXPERIMENT_REPORT.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise HTTPException(503,'IT experiment report is unreadable')

@app.get('/api/it-v2')
def it_v2():
    if not IT_V2_REPORT.is_file():
        return {'available':False, 'reason':'No IT improvement report has been generated.'}
    try:
        report = json.loads(IT_V2_REPORT.read_text(encoding='utf-8'))
        diagnostic = IT_V2_REPORT.parent / 'it_v2_historical_diagnostic.json'
        if diagnostic.is_file():
            report['historical_diagnostic'] = json.loads(diagnostic.read_text(encoding='utf-8'))
        return report
    except (OSError, ValueError):
        raise HTTPException(503,'IT improvement report is unreadable')

class ItPredictionInput(BaseModel):
    text: str = Field(min_length=1, max_length=10000)

def predict_it(text):
    try:
        from ml.it_inference import predict as model_predict
        return model_predict(text)
    except (ImportError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ModelUnavailable(str(exc)) from exc

@app.post('/api/it-predict')
def it_predict(item: ItPredictionInput):
    if not item.text.strip(): raise HTTPException(422,'A message is required')
    try:
        prediction = predict_it(item.text.strip())
    except ModelUnavailable as exc:
        raise HTTPException(503,'IT model is not available yet') from exc
    # Unseen-category rejection is unvalidated; never classify without review.
    return {**prediction, 'needs_review': True}

class PlainTextParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts=[]; self.hidden=0
    def handle_starttag(self, tag, attrs):
        if tag in ('script','style'): self.hidden += 1
        if tag in ('p','br','div','li'): self.parts.append(' ')
    def handle_endtag(self, tag):
        if tag in ('script','style') and self.hidden: self.hidden -= 1
    def handle_data(self, data):
        if not self.hidden: self.parts.append(data)

def import_message(message):
    identifier=message.get('ID')
    if not identifier: return False
    text=message.get('Text') or ''
    if not text.strip() and message.get('HTML'):
        parser=PlainTextParser(); parser.feed(message['HTML']); text=' '.join(' '.join(parser.parts).split())
    if not text.strip():
        text = '[No readable text in this message. Inspect the original in Mailpit.]'
    try:
        return create_ticket((message.get('Subject') or '(No subject)')[:300],text[:10000],'email',message.get('From',{}).get('Address'),identifier) is not None
    except sqlite3.IntegrityError:
        return False

def sync_mail():
    try:
        offset=0; count=0
        while True:
            with urlopen(f'http://127.0.0.1:8025/api/v1/messages?limit=100&start={offset}',timeout=3) as resp: page=json.load(resp)
            summaries=page.get('messages',[])
            if not summaries: break
            with db() as c: known={row[0] for row in c.execute('SELECT mail_id FROM tickets WHERE mail_id IS NOT NULL')}
            for summary in summaries:
                identifier=summary.get('ID')
                if not identifier or not identifier.isalnum() or identifier in known: continue
                with urlopen('http://127.0.0.1:8025/api/v1/message/'+identifier,timeout=3) as resp: message=json.load(resp)
                count+=int(import_message(message))
            offset+=len(summaries)
            if offset>=page.get('total',0): break
        return count
    except (OSError,ValueError) as exc: raise HTTPException(503,'Local Mailpit is unavailable') from exc
@app.post('/api/mail/sync')
def sync(): return {'imported':sync_mail()}
@app.post('/api/test-email')
def test_email(item:TestEmail):
    message=EmailMessage(); message['From']=item.email; message['To']='support@triage.local'; message['Subject']=item.subject; message.set_content(item.body)
    try:
        with smtplib.SMTP('127.0.0.1',1025,timeout=5) as smtp: smtp.send_message(message)
    except (OSError,smtplib.SMTPException) as exc: raise HTTPException(503,'Local SMTP is unavailable') from exc
    try:
        return {'sent':True,'imported':sync_mail()}
    except HTTPException:
        return {'sent':True,'imported':None,'warning':'Message accepted by SMTP; import pending'}

DIST=ROOT/'frontend'/'dist'
if (DIST/'assets').exists(): app.mount('/assets',StaticFiles(directory=DIST/'assets'),name='assets')
@app.get('/{path:path}',include_in_schema=False)
def frontend(path:str):
    if path.startswith('api/'): raise HTTPException(404,'Not found')
    if (DIST/'index.html').exists(): return FileResponse(DIST/'index.html')
    return {'message':'Build frontend to use the app. API docs: /docs'}
