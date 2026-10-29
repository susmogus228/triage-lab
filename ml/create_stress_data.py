"""Author-written synthetic stress fixtures; no customer submissions."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
# Broad labels follow Bitext's peculiar category taxonomy; ambiguous messages require review.
DEV = {
'ACCOUNT': ['My password reset link has expired and I am locked out of my profile.', 'I changed my phone; please help me sign in to my account.', 'cant acces my acount since this morning.', 'The profile page rejects my new login details.'],
'ORDER': ['Please cancel the purchase I made yesterday before it ships.', 'I ordered the wrong size; can you stop order 821?', 'Could you tell me whether my purchase has been confirmed?', 'I want to edit items in an order I placed by mistake.'],
'SHIPPING': ['I entered the old street number. Change my shipping address please.', 'The destination for my parcel is incorrect; edit it before dispatch.', 'Can I update the adress where you send my order?', 'My package needs to go to a different house; change the destination.'],
'CANCEL': ['How much is the early termination fee on my service agreement?', 'What penalty applies if I end this contract before the expiry date?', 'Please explain the charge for cancelling my contract early.', 'I need the cost of ending my service commitment ahead of schedule.'],
'INVOICE': ['Where do I find the tax invoice for last month?', 'Please send a copy of the bill for my purchase.', 'I need the receipt and invoice for accounting.', 'I cant download my invioce; where is it?'],
'PAYMENT': ['Which payment cards can I use at checkout?', 'Does your shop accept bank transfers or only cards?', 'Can I pay with a debit card for this order?', 'My credit card keeps getting declined at payment.'],
'REFUND': ['How can I get my money back after returning the item?', 'My return arrived last week but the refund is missing.', 'What is the policy for reimbursing a damaged product?', 'I still havent received the refund I was promised.'],
'FEEDBACK': ['I am unhappy with your service and want to file a complaint.', 'The agent was rude; where can I submit feedback?', 'I want to register a formal complaint about my experience.', 'Your packaging was awful and I want to share my feedback.'],
'CONTACT': ['How do I speak with a person in customer service?', 'What hours is the support phone line open?', 'Please provide an email address for reaching your support team.', 'Can somebody call me? I cannot find a phone number for support.'],
'DELIVERY': ['Do you ship internationally, and what shipping methods are available?', 'What are the delivery choices for my city?', 'Can you send items by express courier?', 'Which delivery options can I pick when buying?'],
'SUBSCRIPTION': ['Stop sending me your marketing newsletter.', 'How can I unsubscribe from your email newsletter?', 'I want to receive the weekly updates by email.', 'Please remove me from the promotional mailing list.'],
'OUT_OF_SCOPE': ['Can you suggest a recipe for apple pie?', 'My bicycle has a puncture; what tools do I need?', 'What is the capital of France?', 'Write a poem about an invoice drifting in space.', 'How do I clean my aquarium?', 'My cat wants a delivery job; what qualifications are needed?', 'A chess player ordered a bishop to move. Is that legal?', 'Translate the word refund into Japanese.'],
'MULTI': ['I cannot log in and I need a refund for a broken order.', 'Please change my delivery address and email me the invoice.', 'Stop my newsletter and tell me what payment cards you accept.', 'Where is my invoice and can I cancel my purchase?', 'I need the early cancellation fee and want to complain about the agent.', 'I cannot access my account; please also remove me from your mailing list.', 'What shipping methods are available and can I have my money back?', 'I want to contact support and update the parcel destination.']}
TEST = {
'ACCOUNT':['My sign-in code never arrives and I am unable to enter my profile.', 'I get an error whenever I try to log into my account after changing my password.', 'The login page freezes after entering my email.', 'Unable to sign in: my acccount says credentials invalid.'],
'ORDER':['I bought two items by accident. Can you cancel the second purchase?', 'I made a purchase an hour ago and would like to change its items.', 'I want to check whether the order I just placed went through.', 'Stop the order before it is processed please.'],
'SHIPPING':['Please correct the street address for shipment 444.', 'Parcel is going to my former apartment; how do I edit its destination?', 'I typed the wrong post code in the shipping address.', 'Update the address for my purchase to my new place.'],
'CANCEL':['What will you charge me if I terminate this agreement six months early?', 'Is there a penalty for ending the contract before its term is over?', 'How do I find my early termination charge?', 'Need to know cost of breaking the service contract.'],
'INVOICE':['Our bookkeeper needs a VAT bill for the last transaction.', 'Where can I download a copy of my purchase invoice?', 'Need a receipt for my last payment; I cannot find the billing document.', 'Can you resend invoice number 98?'],
'PAYMENT':['Can I use a prepaid card to pay in your store?', 'What payment options are accepted on your site?', 'The checkout says my card is invalid; can you help with payment?', 'Is a wire transfer a valid way to pay?'],
'REFUND':['The returned coat was delivered to you ten days ago and my money has not come back.', 'What conditions must I meet to receive a reimbursement?', 'Please issue a refund for the cancelled purchase.', 'How long before my return is credited to my card?'],
'FEEDBACK':['I want to report how badly I was treated by your employee.', 'Where is the form for complaints about your support?', 'I would like to leave a review of the customer experience.', 'The service has been frustrating and I want to file a complaint.'],
'CONTACT':['Could you provide customer service opening hours?', 'Is there a telephone number to reach a support representative?', 'I would like to speak to an agent; where can I contact you?', 'Please tell me how to get in touch with the help desk.'],
'DELIVERY':['Can I choose next-day delivery for this location?', 'Which countries do you ship orders to?', 'Do you offer standard and express shipping services?', 'What courier options are available in my area?'],
'SUBSCRIPTION':['Take my email off your weekly promotions list.', 'How do I sign up for the company newsletter?', 'Please stop the advertising emails from your business.', 'I want to unsubscribe from marketing updates.'],
'OUT_OF_SCOPE':['What can I cook with mushrooms and rice?', 'When is the next lunar eclipse?', 'How do I train my puppy to sit?', 'Can an airplane deliver an invoice to Mars?', 'Is it possible to refund a chess move?', 'I ordered a dog to sit but it ran away.', 'Please translate my shipping forecast into French.', 'My bike needs a new tyre; recommend a repair tool.'],
'MULTI':['Reset my password and refund my last order.', 'I want to report rude service and unsubscribe from emails.', 'Please change the shipping address and tell me which payment cards you accept.', 'Can you send an invoice and cancel the purchase I just made?', 'I cannot sign into my account and need the support phone number.', 'What are the delivery methods and how do I request a refund?', 'Tell me the contract termination fee and stop the newsletter.', 'The invoice is missing and my shipping destination is incorrect.']}
def write(split,groups):
 rows=[]
 for label,texts in groups.items():
  for n,text in enumerate(texts,1):
   rows.append({'id':f'{split}-{label.lower()}-{n:02}', 'text':text,'expected':label,'kind':'out_of_scope' if label=='OUT_OF_SCOPE' else 'multi_intent' if label=='MULTI' else 'in_scope', 'provenance':'assistant-authored synthetic', 'split':split})
 path=ROOT/'data'/f'stress_{split}.json';path.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(path,len(rows))
if __name__=='__main__':
 write('dev',DEV);write('test',TEST)
