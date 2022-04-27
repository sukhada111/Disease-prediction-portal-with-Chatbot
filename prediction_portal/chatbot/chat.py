import random
import json

import torch

from chatbot.model import NeuralNet
from chatbot.nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
path=os.getcwd()+'\\chatbot\\intents.json'
with open(path, 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = os.getcwd()+"\\chatbot\\data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Healthino"

from chatbot.sentiment_model import predict_custom_tag, predict_custom_tweet_sentiment

response_dict={

     'workplace_anx':
           ["Sad to hear that this happened. However, life has ups and downs and you are not the only person who  might have to face this situation. Instead of wasting time by repenting upon incorrect decisions or bad luck, focus on the future and what positives can be drawn.", "You must try communicating with your colleagues and seniors. Start speaking out your feelings.", "Things will get easy, just have patience. Be persistent with your work and hard work, don't focus on things that are not in your control and soon the problems will disappear.","It is important for you not only to work, but also focus on your mental health.","Don't worry much regarding this, it will all be okay in some time. Besides dedication, one needs to have positive and optimistic outlook towards things he does."],

       'workplace_pos':
           ["Excellent! It seems you have mastered your field","Great to hear this. I think there's more to come","Being a mental health assistant and chatbot, I didn't expect that you would share something positive, but that surely makes me proud of your work and I hope you stay consistent at this level " ,"Helping different people everyday kind of makes me bored and negative, but now i got something to rejoice!!", "Today it might be your day, but always remember to put a smile on your co-workers' and juniors' faces too! Congratulations!"  ,"Don't forget to appreciate and thank those who contributed and helped you in making this task successful!"],

     'workplace_depr' : ["That's Sad! We all have good, bad, ugly and worse days in our lives. Consider this as one of the worse days and move on. Remember that for being successful, one needs to overcome his own failures and fears.", "Sorry to hear that. However, consider this as a bad luck. Things always happen for good and you must not be thinking regarding this as a failure, who knows what's kept on the other side.", "You must not start feeling demotivated to depressed, instead share your thoughts with your colleagues and bosses, I am sure, they would respect them.", " one always has to fight for success. Continue your struggle and you might be there one day", "I think you are stressing too much on your work. Remember that work is just a part of your life, not your life. Your body, family and other responsibilities deserve as much attention as your work. Try and prioritise on things. These problems in work would get solved one day or another, but problems and diseases caused by mental health issues can be irreversible."],

     'workplace_sad': [ "Don't worry, we all have bad days. But the positives after all this is that there is always a positive tomorrow that we all look forward to.", "It's the happy things that people always look forward too in sad times, similarly your time would also come where you would be shining as a start in the darkness of the sky ","Don't worry, the work environment would heal gradually. However, it won't be justified to neglect other important things like your health and important family time. One must always keep their professional and personal lives separate. Whatever happened, you must not forget the happiness of your loved ones.", "Try spending your time by practising any hobby, take your family out for a dinner or just let your kids or pets be around you! Time will heal all the pain and you would be fresh again to look forward to your professional goals."],


       'relationship_anx':
           ["Communication is the key for a happy relationship. Tell your partner and show that you love them. Let them know that you understand that fights are affecting their thoughts, feelings, and behavior and that you love them. Reassure them that you are here to support them in their journey to get better.", "Discuss with your partner the different reasons for them to stay. Perhaps it's the dependents, a beloved pet who needs them, or their faith. These reasons can help them hold on a bit longer until the pain subsides.","Give your partner the required space after an argument and let things settle down between the two of you. Concentrate on other activities and take time off each other. "],

       'relationship_pos':
           ["That's wonderful to know that things are working fine for the two of you. Healthino wishes you the best for a great future ahead!", "Only a few relationships are as good as yours. Have a great time together and cheers to all the happy memories.", "I think you guys were destined to be together for your life-time. Healthino wishes to find a relationship just as perfect as yours!", "A healthy relationship is all about balance and chemistry. Healthino identifies true love and wishes that your love lasts forever!!"],

      'relationship_depr': ["Sharing your feelings with friends and family will help you get through this period. Join a support group where you can talk to others in similar situations. Isolating yourself can raise your stress levels, reduce your concentration, and get in the way of your work, other relationships, and overall health.", "Talk about your feelings with your close ones. Knowing that others are aware of your feelings will make you feel less alone with your pain and will help you heal. Writing in a journal can also be a helpful outlet for your feelings." , "Expressing your feelings will liberate you in a way, but it is important not to dwell on the negative feelings or to over-analyze the situation. Getting stuck in hurtful feelings like blame, anger, and resentment will rob you of valuable energy and prevent you from healing and moving forward. Encourage yourself by the fact that new hopes and dreams will eventually replace your old ones.", "Help yourself heal by scheduling daily time for activities you find calming and soothing. Spend time with good friends, go for a walk in nature, listen to music, enjoy a hot bath, get a massage, read a favorite book, take a yoga class, or savor a warm cup of tea." ],
     'relationship_sad': ["Always remember that don't take decisions when you are sad or angry regarding something. Every relationship has its ups and downs. It's only the fight you put that keeps the relationship alive.", "A relationship is never about 'I', it's always about 'us'. Listen to your partner, understand their opinions on the situations and work things out together.", "Let your partner know that you love them, you respect them and you care about them. Communicate your feelings on a regular basis. Healthino believes that love and effort overcome every obstacle!"],


    'family_anx': ["While we would all prefer to get along well with our families, it is not always possible to do so. Generally people rely a lot on families for mental support however in cases like these, you need to talk it through with a friend or a therapist and figure out ways to iron out the differences between you and your family because there is always light at the end of the tunnel.", "It is not always possible to have happy times with our families. There will be times when you face differences and it is completely normal since every person has their own opinion and outlook on life. However what is important is to sort the issues out and bring back the happy moments. If you need someone to share it with, I am always there." , "Sometimes it is important to think from another person's point of view. Our priorities may shift suddenly in a crisis. Make sure you understand and honor the needs of family members or other household residents during the recovery process.", "Almost all of us face tough times with family at least once in our life. In such situations, take time to do something that is meaningful, relaxing and fun to you and your family. Read a book, sit on the porch and enjoy the scenery, enjoy coffee, or have a family movie or game night. These little things will definitely help you get over it."],

'family_pos': ["Happy to hear that you and your family are doing well. Wishing you happy times ahead.", "I am glad that you are in a good mindspace. Family support always helps us stay happy!",  "So happy to know that you and your family are putting a smile on each others' face. There is only better to come.",  "Amazing! I hope you have the best time in the coming days.", "I am extremely happy for you and I hope you have exciting times ahead!" ],

'family_depr': ["Family issues are commonly experienced by many people but intensifying them to an extent where it takes a toll on your mental health is not normal. Either the issue needs to be solved by clearing out things with family or if it is a persistent issue and you know that it won't be solved easily, then opening up about it to a therapist would be better. Take enough sleep and do exercise or yoga to improve your mental state", "Negative people are like a parasite to the soul. They can eat up the willingness to stay happy and when this negativity comes from family, the situation gets worse. As much as we want to be there for our family, it is important you focus on your mental health too. Try doing certain activities alone, away from the source of negativity in your life. Take a walk, go on a trek, try out solo dinners. It will help you heal a little.", "At times when the situation is not in our control, we need to focus on things that are in our control and that is our own self. It is important to become the hero of your story and not let external factors affect your mental health and self confidence. You always have the option to discuss severe problems with a therapist but focusing on your well being is extremely necessary." ],


'family_sad': ["Every family has to go through some ups and downs, it is important to hold on to each other and figure a way out together. I hope you and your family will overcome all the difficulties soon.", "Tough times are unavoidable and unforeseen at times. You need to be strong and keep your loved ones in a good mental space as well. I wish you all the strength to get over this hurdle.", "Just like happy days, we all have sad days too. In the end our biggest support system is our family and tough times will only strengthen your love and the bond that you share.", "As they say, Families are like a fudge, mostly sweet but with a few nuts. I understand the situation is tough for you and your family but I am sure you will get over it together and be there for each other through it all." ],

'friends_sad':[ " I understand. Friends are indeed an important part of a person's life. And that is exactly why you should keep trying to mend the issues and not give up on each other during the ups and downs. Communication, along with having compassion is the key.  ", 
" I get it, having a friend beside you is what gets you through all the bad times. Perhaps, you should try understanding why you/your friend acted like that and identify the root cause in order to fix the issue. Empathy is very important to establish communication and sort things out. ",
"That's sad. However, a friendship with a strong foundation can always be reignited if efforts are taken in the right direction to mend the problems. ",
"Don't worry, a true friend always remembers the good and the bad times spent together and will eventually revive your friendship because it means a lot to them as well."
 ],

'friends_anx': [ " I'm sad to know that and I hope things get better soon, perhaps sending a surprise apology or something that reminds of the great times you have spent together, is a thing that will melt the toughest heart and pave the way for reviving your friendship. ",
"Sorry to hear that. A friend is someone who makes good times better and bad times easier and it can be difficult to not have your go-to person with you. But, it is also true that, if you take that first step towards fixing your friendship, it will surely do some good and you won't have to regret not fighting for your friendship and perhaps, your friend is expecting this effort from you. ",
"Oh, I'm sad to know that. But it's alright, all friendships go through the ups and downs but at the end, if the friendship is deep enough, it will always turn out okay and there's nothing a little friendly chat or a sweet, lovely gift can't fix. "
],

'friends_depr':[ " Oh, that's awful. I understand that it must be hard considering the bond you must have with your friend all this time and that it may look like things aren't going to be alright. But, trust me, time heals everything. Don't lose hope and things will get sorted out soon.",
" That's unfortunate. Perhaps, all you need is a little bit of personal time to distract yourself and get it off your mind. Plus, there are always other people around you who would be just as lovely a friend to you and they can help you out. ",
"I understand, friendships can be difficult. Especially when it's with the person whom you run to when you're sad or facing a problem. But, it's not the end of the World. You can always spend some time with your family and close ones to make yourself feel better and get a much-deserved break from all of it. With time, the hurt will ease and you will feel better.  ",
" It's not easy but things will get better. Perhaps, if it's hurting so much it is time to take a step back for yourself and reevaluate your friendship. After all, you deserve someone who understands you and is equally glad to have you as their friend. Spend some quality alone time pursuing your hobby or listening to your favourite songs. "
],

'friends_pos': [ "That's great. I am glad things are going well because it's important to have a good friend circle around yourself!",
" Glad to hear that! A good friend is what gets you through the tough times and makes your life happier and better in so many ways!",
"A great friendship is something we all need and I am glad you have such nice companions throughout your life's journey."
 "You really are a blessed person to have such great and lovely friends by your side. Hope things will just get better with time!"
],

'financial_anx':["Hey, financial problems tend to impact the whole family and enlisting your loved ones' support can be crucial in turning things around. Even if you take pride in being self-sufficient, keep your family up to date on your financial situation and how they can help you save money.", "Hey, tracking your finances in detail can also help you start to regain a much-needed sense of control over your situation. Detail your income, debt, and spending over the course of at least one month. A number of websites and smartphone apps can help you keep track of your finances moving forward or you can work backwards by gathering receipts and examining bank and credit card statements.","Hey to deal with your financial problems, plan a monthly budget. Setting and following a monthly budget can help keep you on track and regain your sense of control. Feeling tired will only increase your stress and negative thought patterns. Finding ways to improve your sleep during this difficult time will help both your mind and body."],

'financial_pos':["Healthino is delighted to know that you are doing great financially! Keep working hard and put equal efforts to prosper in life!", "Hey, happy to know that you are financially stable in life. Invest your money with utmost care, just like you have been doing and keep prospering in life!"],

'financial_depr': ["When you're facing money problems, there's often a strong temptation to bottle everything up and try to go it alone. You may feel awkward about disclosing the amount you earn or spend, feel shame about any financial mistakes you've made, or embarrassed about not being able to provide for your family. But bottling things up will only make your financial stress worse. In the current economy, where many people are struggling through no fault of their own, you'll likely find others are far more understanding of your problems.", "Hey! I see you are facing financial issues. Take inventory of your financial situation and eliminate discretionary and impulse spending. Address your problem and make a plan to live within a tighter budget, lower the interest rate on your credit card debt, curb your online spending, seek government benefits or find a new job or additional source of income. Following these things will help you overcome your mental stress and provide financial stability." , "Hey, resolving financial problems tends to involve small steps that reap rewards over time. In the current economic climate, it's unlikely your financial difficulties will disappear overnight. But that doesn't mean you can't take steps right away to ease your stress levels and find the energy and peace of mind to better deal with challenges in the long-term. Take time to relax each day and give your mind a break from the constant worrying. Meditating, breathing exercises, or other relaxation techniques are excellent ways to relieve stress and restore some balance to your life."],
'financial_sad': ["Hey I see you are facing financial issues. Speaking with a free financial counselor may ease your mind about your debt or approaching a career counselor can also be a good resource if you're looking for that next step in your job path. Do things that make you happy and will keep your mind away from the negative thoughts.", "Hey, when you're plagued by money worries and financial uncertainty, it's easy to focus all your attention on the negatives. While you don't have to ignore reality and pretend everything's fine, you can take a moment to appreciate a close relationship, the beauty of a sunset, or the love of a pet, for example. It can give your mind a break from the constant worrying, help boost your mood, and ease your stress.", "Hey, experiencing financial problems can impact your self-esteem. But there are plenty of other, more rewarding ways to improve your sense of self-worth. Even when you're struggling yourself, helping others by volunteering can increase your confidence and ease stress, anger, and anxiety—not to mention aid a worthy cause. You could spend time in nature, learn a new skill, or enjoy the company of people who appreciate you for who you are, rather than for your bank balance."],
     
'social_anx': ["To be honest, anxiety due to social media is very common especially because of Fear of Missing Out and feeling of inadequacy but remember that it is not always how it looks. So it is important to do things that actually make you happy rather than social media standards.", "Look I understand that you might be feeling that you do not ‘fit in' the fickle world of social media by looking at others' lifestyles. However, do not use social media to compare your life with others. Instead use it to lighten your mood and just as a source of entertainment and connecting with friends.", "Looks like you are attaching too much importance to a dumb thing like social media. I understand it helps you feel connected to the world but it should stay limited to that. Do not let it get to you and affect your mood and mind. It's always better to detach from things that ruin your mental peace" ],

'social_pos': ["Happy to know that social media is helping you to lighten your mood and be in a happy mental space", "Hey it's amazing that you are happy and using social media for a positive purpose. It really has become a huge platform to connect people and generate new income sources.", "I am so glad that you are doing well and are enjoying your time on social media. It can get a little too much sometimes so it is important to balance the usage properly."],

'social_depr': ["Hey I really think you should take a break from social media. It can be a very toxic place to be at times and also make you feel highly dissatisfied about your life. Many people suffer social anxiety as a result. Also this platform is very prone to cyber bullying, judgements and difference of opinions and hence it is important that you do not let these things get under your skin and learn to value the real things in life.","Hey it'll be fine, take a deep breath first. It is said that the less time you spend on social media, the more the qualitative shift in your life.Do not treat it like the testament of your social life and most importantly stop comparing your life with what you see on social media. It is often fake and can make you feel lonely which will affect your mental peace. Treat social media like an optional source of leisure and do things that make you genuinely happy.", "I understand. Social media can be a very negative space to be in since it exposes you to much sensitive content, chances of getting bullied by complete strangers and most importantly, it can create an image that will make you feel dissatisfied about your life. It has its good side but it has made us more worried about what people will think about us and worry about things that do not even matter in real life." ],


'social_sad':["Hey don't worry, this feeling is just temporary and I am sure if you try thinking less about it and focus more on activities that'll make you happy, you will soon forget it. Don't think so much about something like social media which is so fickle and ignore the negativity.", "Hey this shall definitely pass and the only thing you should focus on is not letting social media affect your mind negatively. The moment you feel sad because of social media, I suggest you try to reduce your activity time and engage more in refreshing activities.", "Hey it's completely okay to feel this way. We all have our good and bad days and just like that social media can also be good and bad at the same time. The ultimate control as to how to let it affect our mind lies in our control and hence we must balance that properly. " ],

 'career_anx':
           ["That's alright. At some point or other, we all face hurdles in our career but know that, whenever a door closes, another opens and there are plenty of opportunities out there if you simply stay hopeful and look for it.",
"Don't worry, it will be fine soon. Sometimes, all you need to do is look on the bright side and keep upgrading your skills and profile to enhance your career.",
"Ohh, but don't lose heart. After all, you have developed skills over the years and they are never going to go to waste. Perhaps, connect to some people working in your industry through platforms such as LinkedIn who can guide you further."
],

'career_pos':
["That's great! Keep going, Sky's the limit, never doubt yourself, stay focused, don't let anything stop you from conquering your goals and making your dreams happen!",
"Glad to hear that! Wishing you more and more success in taking your career to greater heights!", 
"That's amazing. Hard work always pays off. But this is just the start, I'm sure great things are coming your way!",
" Woww, glad to know you are pursuing your dream career and achieving great levels of success!"
],

      'career_depr': [ "Sorry to hear that. I know it must be difficult for you. A career is something we work on for our entire life but there are always new options to explore. Do not limit yourself, it's never too late to pursue something new and exciting. " ,
 "That's unfortunate. But remember, Opportunities don't happen, you create them. So don't let it affect you this much, try upgrading your skills or talking to someone more experienced to help you find the right direction.  ",
 "Don't worry. We have all been there. But perhaps, it is time to find and pursue what you are really passionate about and learn the necessary skills to launch your career forward.",
 "Building a good career can be challenging. But don't limit yourself. Many people limit themselves to what they think they can do. You can go as far as your mind lets you. What you believe you can achieve. So explore new opportunities. ",
"Everyone has to face some obstacles before finally achieving the much desired goals and the key is to stay calm, believe in yourself and keep working towards it. Remember, where there's a will there'a a way."

],
     'career_sad': ["That's okay. Being on the career path that you have chalked out for yourself can be tough but it will get better with time. Hard work and dedication always pays off.",
"Sad to know that. But, hey you can always look for new domains in your field or upskill yourself or even pursue a new degree to get a fresh outlook about your career. After all, you chose it because it's what you love doing. ",
"Life is tough, but nothing gives more happiness than pursuing something you love even if it's difficult for a while.",
"It'll be okay soon. Meanwhile, try to build connections with more experienced individuals and keep yourself occupied so that when an opportunity does come, you will be ready for it."
]


}

def get_response(msg):
    input_msg=msg
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    # max_word_len=394
    # X= predict_custom_tag(sentence,max_word_len)
    # print(X.shape)
    X = X.reshape(1, X.shape[0])
    # print(X.shape)
    X = torch.from_numpy(X).to(device)
    # X = torch.Tensor(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.7:
        for intent in intents['intents']:
        
            if tag == intent["tag"]:
                print(tag)
                context_tags=['family','relationship', 'friends', 'workplace', 'social', 'financial', 'career']
                if tag in context_tags:
                
                    sentiment=predict_custom_tweet_sentiment(input_msg)
                    print(sentiment)
                    if sentiment>0.5:
                        level='pos'
                      
                    else:

                        if sentiment<=0.1:
                            level='depr'
                           
                        elif sentiment>0.1 and sentiment<=0.3:
                            level='anx'
                        
                        elif sentiment>0.3 and sentiment<=0.5:
                            level='sad'
                        
                    op_tag=tag+'_'+level
                    return random.choice(response_dict[op_tag]), tag
                
                else:
                    #its an info tag
                    print(tag)
                    sentiment=predict_custom_tweet_sentiment(input_msg)
                    print(sentiment)
                    return random.choice(intent['responses']), tag
    else:
        s = "I do not understand..."
        return s, "not found"

  