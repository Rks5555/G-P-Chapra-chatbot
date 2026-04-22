
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
client = OpenAI(api_key="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")

# Hinglish normalization dictionary
hinglish_map = {
"kaise": "how",
"kya": "what",
"kab": "when",
"kahan": "where",
"hai": "",
"h": "",
"ka": "",
"ke": "",
"ki": "",
"kitna": "fee",
"fees": "fee",
"hostel": "hostel",
"admission": "admission",
"process": "process",
"college": "college"
}

# Questions & Answers
qa_pairs = {
"hi": "Hello! How can I help you?",
"Hii": "Hello! How can I help you?",
"hlo": "Hello! How can I help you?",
"H": "Hello! How can I help you?",

"Kya": "Hi there! Welcome to GP Chapra chatbot.",
"hello": "Hi there! Welcome to GP Chapra chatbot.",
"Hello": "Hi there! Welcome to GP Chapra chatbot.",
"what is diploma qualification": "Diploma ek technical course hota hai jo 10+2 ke level ke barabar maana jata hai, Isme practical aur technical skills pe zyada focus hota hai.",
"what can i do after diploma": "Diploma ke baad aap job ke liye apply kar sakte ho ya phir lateral entry ke through B.Tech/B.E. me admission le sakte ho.",
"diploma ka fayda ": "You can apply for SSC ,SSC JE, RRB JE, DRDO, Clerk , officer jobs ,junior engineer and many more .",
"diploma k baad job ": "You can apply for SSC ,SSC JE, RRB JE, DRDO, Clerk , officer jobs ,junior engineer and many more .",
"Polytechnic karne ka fayda ": "You can apply for SSC ,SSC JE, RRB JE, DRDO, Clerk , officer jobs ,junior engineer and many more .",
"diploma k baad kya kre ": "You can apply for SSC ,SSC JE, RRB JE, DRDO, Clerk , officer jobs ,junior engineer and many more .",
"diploma ka benefits ": "You can apply for SSC ,SSC JE, RRB JE, DRDO, Clerk , officer jobs ,junior engineer , B.Tech and many more .",
"college info": "Government Polytechnic Chapra ek government diploma engineering college hai jo Marhaura, Saran (Bihar) me located hai aur yaha students ko technical education, practical knowledge aur placement opportunities milti hain.",
"Kuch btao": "Like what, addmission , branches, campus, exam.",
"gp chapra detail": "GP Chapra ek government diploma engineering college hai jo Marhaura me hai aur yaha technical education aur placement opportunities milti hain.",

"where college located": "GP Chapra is located in Marhaura, Saran district, Bihar.",
"college location": "GP Chapra is located in Marhaura, Saran district, Bihar.",
"college location": "GP Chapra is located in Marhaura near main market. There is a block office in front, a high school, petrol pump and government ITI nearby. It is about 25 km from Chapra.",
"college kaha hai": "GP Chapra Marhaura me main market ke paas located hai.",

"gp chapra kaha hai": "GP Chapra Marhaura me hai jo main market ke pahle hi hai.",
"college exact location kya hai": "College Marhaura me main market ke pahle hai jiske samne block office hai.",

"college ke aas paas kya hai": "College ke paas block office, high school, petrol pump aur government ITI hai.",
"nearby places kya hai": "Nearby places me block office, school, petrol pump aur ITI include hain.",

"college market ke paas hai kya": "Yes, college main market ke bahut paas located hai.",
"main market se kitna dur hai": "College main market ke pahle hi located hai, bahut paas hai.",

"chapra se kitna dur hai": "College Chapra se approx 25 km dur hai.",
"distance from chapra": "GP Chapra is about 25 km away from Chapra.",

"location details": "GP Chapra Marhaura me located hai near main market with nearby block office, school, petrol pump and ITI, about 25 km from Chapra.",

"admission process": "Admission is done through DCECE Polytechnic entrance exam.",
"how take admission": "Admission is done through DCECE Polytechnic entrance exam.",
# ------------------ DOCUMENT COPIES & PHOTOS ------------------

"documents kitni copies chahiye": "Admission ke liye har document ki 2-3 copies honi chahiye.",
"documents copies kitni lagegi": "Students ko har document ki 2-3 photocopies ready rakhni chahiye.",

"passport photo kitna chahiye": "Admission ke liye minimum 10 passport size photos chahiye hote hain.",
"photos kitne chahiye admission me": "Students ko kam se kam 10 passport size photos rakhna chahiye.",

"admission ke liye kitne documents chahiye": "Har document ki 2-3 copies aur 10 passport size photos ready rakhna chahiye.",
"admission me kya kya ready rakhe": "Documents ki photocopies (2-3 each) aur minimum 10 passport size photos ready rakhna chahiye.",

"document copies kyu chahiye": "Photocopies verification aur record ke liye use hoti hain isliye multiple copies zaruri hoti hain.",
"photo kyu chahiye admission me": "Passport size photos form filling aur official records ke liye use hote hain.",

"admission tips": "Admission ke time har document ki 2-3 copies aur kam se kam 10 passport size photos zarur le kar aayein.",
"important admission advice": "Always carry multiple document copies and passport size photos during admission process.",
# ------------------ CLC REQUIREMENT ------------------

"clc kya hai": "CLC (College Leaving Certificate) ek important document hai jo admission ke liye zaruri hota hai.",
"clc full form kya hai": "CLC ka full form College Leaving Certificate hota hai.",

"clc kyu important hai": "CLC admission ke liye sabse important document hota hai.",
"clc zaruri hai kya": "Yes, CLC ke bina admission possible nahi hai.",

"admission ke liye clc chahiye kya": "Haan, admission ke liye CLC hona compulsory hai.",
"clc ke bina admission ho sakta hai kya": "Nahi, CLC ke bina admission bilkul possible nahi hai.",

"clc kaha se milta hai": "CLC aapke previous school ya college se milta hai.",
"clc kaise milega": "Students apne previous school/college se CLC lete hain.",

"admission important document": "Admission ke liye CLC sabse important document hota hai.",
"important document for admission": "CLC is the most important document required for admission.",

"clc details": "CLC (College Leaving Certificate) is mandatory for admission and must be obtained from previous school or college.",


"fee structure": "Fees are around ₹3000 to ₹5000 per semester.",
 
# ------------------ HOSTEL & MESS FEES ------------------

"hostel fee": "Hostel fee approx ₹4500 (old hostel) aur ₹6500 (new hostel) per semester hota hai.",
"hostel fee kitna hai": "Old hostel ka fee ₹4500 aur new hostel ka ₹6500 per semester hota hai.",

"old hostel fee": "Old hostel ka minimum fee approx ₹4500 per semester hota hai.",
"new hostel fee": "New hostel ka minimum fee approx ₹6500 per semester hota hai.",

"mess fee": "Mess ka kharcha approx ₹2000–3000 per semester hota hai.",
"mess fee kitna hai": "Mess ke liye lagbhag ₹2000–3000 lagta hai per semester.",

"hostel aur mess total fee": "Hostel aur mess mila kar total approx ₹7000–9000 per semester lagta hai.",
"total hostel mess cost": "Combined cost hostel + mess ka lagbhag ₹7000–9000 per semester hota hai.",

"hostel me rehne ka kharcha": "Hostel aur mess ka total kharcha ₹7000–9000 per semester ke around hota hai.",
"hostel cost kitna hai": "Total cost hostel aur mess mila kar approx ₹7000–9000 hota hai.",

"hostel options": "College me old aur new dono type ke hostel available hain different fee structure ke sath.",
"kitne hostel hai": "College me old hostel aur new hostel dono available hain.",

"hostel details": "Old hostel ₹4500, new hostel ₹6500 aur mess ₹2000–3000 per semester hota hai, total ₹7000–9000 ke around.",
# ------------------ COURSE STRUCTURE ------------------

"course duration": "Diploma course 3 years ka hota hai.",
"course kitne saal ka hota hai": "Yeh course total 3 saal ka hota hai.",

"kitne semester hote hai": "Total 6 semester hote hain 3 saal me.",
"total semester": "Course me total 6 semesters hote hain.",

"1 saal me kitne semester hote hai": "1 saal me 2 semester hote hain.",
"per year semester": "Har saal 2 semesters conduct hote hain.",

"semester duration": "Har semester approx 4–6 months ka hota hai.",
"semester kitne months ka hota hai": "Ek semester lagbhag 4 se 6 mahine ka hota hai.",

"exam kab hota hai": "Har semester ke end me exam hota hai.",
"semester exam hota hai kya": "Yes, har semester ke baad exam conduct hota hai.",

"exam pattern": "Students ko har semester ke end me exam dena hota hai.",
"har semester exam hota hai kya": "Haan, har semester ke end me exam hota hai.",

"course structure": "Course 3 saal ka hota hai jisme 6 semester hote hain, har saal 2 semester aur har semester 4–6 months ka hota hai.",
"course details": "Diploma course me 3 years, 6 semesters aur har semester ke end me exam hota hai.",
# ------------------ FEES DETAILS ------------------

"admission fee": "Admission fee approx ₹1000–1500 hota hai, jo category ke basis par vary karta hai.",
"admission fee kitna hai": "Admission fee lagbhag ₹1000–1500 hota hai (category ke according).",

"har saal fee lagta hai kya": "Yes, admission fee har saal lagta hai.",
"yearly fee kitna hota hai": "Har saal approx ₹1000–1500 admission related fee lagta hai.",

"exam fee": "Exam fee bhi approx ₹1000–1500 hota hai depending on category.",
"exam fee kitna hota hai": "Exam fee lagbhag ₹1000–1500 hota hai (category ke basis par).",

"carry exam fee": "Carry (backlog) paper ka exam fee bhi approx ₹1000–1500 hota hai.",
"backlog fee kitna hota hai": "Backlog ya carry paper ke liye bhi ₹1000–1500 fee lagta hai.",

"carry paper fee": "Carry paper exam ke liye bhi same range ka fee lagta hai (₹1000–1500).",
"backlog exam fee": "Backlog exam fee bhi category ke hisab se ₹1000–1500 hota hai.",

"fees category pe depend karta hai kya": "Yes, fees caste/category ke basis par thoda vary karta hai.",
"category wise fee": "Fee structure category ke according change hota hai.",

"total fee details": "Admission, exam aur carry paper sabka fee approx ₹1000–1500 hota hai depending on category.",
# ------------------ SCHOLARSHIP ------------------

"scholarship": "College provides scholarship every year where students get around ₹3000–4000.",
"scholarship milta hai kya": "Yes, college me har saal scholarship milta hai students ko.",

"scholarship kitna milta hai": "Students ko approx ₹3000–4000 tak scholarship milta hai.",
"scholarship amount kya hai": "Scholarship ka amount lagbhag 3000 se 4000 rupaye hota hai.",

"scholarship kisko milta hai": "Scholarship girls aur boys dono students ko milta hai.",
"girls ko scholarship milta hai kya": "Yes, girls ko bhi scholarship milta hai.",
"boys ko scholarship milta hai kya": "Yes, boys ko bhi scholarship milta hai.",

"scholarship ke liye kya karna hota hai": "Students ko online form fill karna hota hai aur registration office me submit karna hota hai.",
"scholarship form kaise fill kare": "Scholarship ke liye online form fill karke office me submit karna hota hai.",

"scholarship ke liye documents kya chahiye": "Scholarship ke liye multiple documents aur teacher ke signature required hote hain.",
"documents for scholarship": "Students ko scholarship ke liye documents aur teacher ka sign submit karna hota hai.",

"teacher help karte hai kya scholarship me": "Yes, teachers students ko scholarship process aur form filling me guide karte hain.",
"scholarship info kaise milti hai": "Scholarship ki sari information teachers provide karte hain.",

"scholarship submit kaha hota hai": "Scholarship form registration office me submit kiya jata hai.",
"scholarship kaha jama kare": "Form fill karne ke baad registration office me submit karna hota hai.",

"scholarship details": "Students get yearly scholarship of around ₹3000–4000. Form is filled online and submitted to office with required documents and teacher guidance.",



"hostel available": "Yes, hostel facility is available.",

"courses offered": "Courses include CSE, Civil, Electrical, Mechanical, Automobile, Electronics Engineering etc.",
# placement
"placement": "Placements in GP Chapra are strong especially for Electrical, Electronics, Automobile and Mechanical branches. Many companies visit and offer internships and training with salary starting around ₹15000 per month.",
"placement details": "Placements are strong for Electrical, Electronics, Automobile and Mechanical branches. Companies offer internships (6 months) and training (1 year) with stipend around ₹15000 or more.",
"placement kaisa hai": "Placements are good for Electrical, Electronics, Automobile and Mechanical branches. Many companies visit with internship and training offers.",
"college placement kaisa hai": "Placement is strong for core branches like Electrical, Electronics, Automobile and Mechanical. Companies offer stipend around ₹15000+.",

"which branch has best placement": "Electrical, Electronics, Automobile and Mechanical branches have better placement opportunities.",
"best placement branch": "Electrical, Electronics, Automobile and Mechanical branches have strong placements.",

"cs placement": "Placement for Computer Science is comparatively lower, but some companies do visit. Many students go for higher studies.",
"civil placement": "Placement for Civil branch is limited compared to core branches, but companies do visit sometimes.",

"cs placement kaisa hai": "CS placement is comparatively lower, but companies do visit. Students often prefer higher studies.",
"civil placement kaisa hai": "Civil placement is limited but some companies visit campus.",

"kitni salary milti hai placement me": "Students can get around ₹15000 or more per month during internship or training.",
"placement salary": "Placement salary or stipend usually starts from ₹15000 per month.",

"internship milta hai kya": "Yes, companies offer 6 months internship and 1 year training programs with stipend.",
"training milta hai kya": "Yes, companies provide training programs along with stipend.",

"companies aati hai kya": "Yes, many companies visit campus especially for core branches like Electrical, Mechanical, Automobile and Electronics.",
"campus placement hota hai kya": "Yes, campus placement is available. Many companies visit for core branches.",

"higher studies after diploma": "Many CS and Civil students prefer higher studies due to limited placement opportunities.",


# id card
"id card policy": "Students are issued a college ID card after admission. It is mandatory to carry it daily in college.",
"id card rules": "Students must carry their ID card every day. It is an important identification proof inside the campus.",

"id card kab milta hai": "Students get their ID card after taking admission and starting college.",
"id card kaise milta hai": "Students have to provide their personal details, then teachers issue the ID card.",

"id card compulsory hai kya": "Yes, it is compulsory to carry ID card daily in college.",
"id card lana zaruri hai kya": "Yes, har din college aate time ID card lana zaruri hai.",

"id card ka use kya hai": "ID card is used as identification proof inside the college campus.",
"id card kyu important hai": "ID card college ke andar identification proof ke liye important hota hai.",

"id card kho gaya to kya kare": "Students should take care of their ID card. It should not be lost or misplaced.",
"id card lost ho gaya to": "ID card ko safe rakhna zaruri hai. Agar lost ho jaye to college se contact karna chahiye.",

"id card safety": "Students must take proper care of their ID card and should not lose it.",
# uniform
"uniform policy": "College has a proper uniform with black and white dress code for all students.",
"uniform rules": "Students must wear proper uniform. Only formal dress is allowed in the campus.",

"uniform kaisa hai": "College uniform follows black and white dress code for both boys and girls.",
"college uniform kya hai": "Uniform includes black and white dress code with proper formal attire.",

"girls uniform kya hai": "Girls wear white suit with black salwar and black dupatta.",
"ladkiyon ka uniform kya hai": "Girls ka uniform white suit, black salwar aur dupatta hota hai.",

"boys uniform kya hai": "Boys wear white shirt, black pants, tie and formal shoes.",
"ladko ka uniform kya hai": "Boys ka uniform white shirt, black pant, tie aur formal shoes hota hai.",

"uniform compulsory hai kya": "Yes, wearing uniform is compulsory in college.",
"uniform pehenna zaruri hai kya": "Yes, college me uniform pehenna zaruri hai.",

"casual dress allowed hai kya": "No, casual or inappropriate clothes are not allowed in college.",
"college me casual kapde allowed hai kya": "Nahi, sirf formal uniform hi allowed hai.",

"dress code kya hai": "College follows strict black and white formal dress code.",
"dress code rules": "Only formal uniform is allowed. Casual clothes are not permitted.",

# faculty
"faculty": "College has many knowledgeable and supportive professors who help students in academics and overall development.",
"faculty kaisa hai": "Faculty is very good, friendly and supportive. Teachers are always ready to help students.",

"teachers kaise hai": "Teachers are knowledgeable, friendly and supportive towards students.",
"professors kaise hai": "Professors are highly experienced and always help students in studies and guidance.",

"kitne teachers hai": "Each branch has 3 or more teachers assigned to guide students.",
"har branch me kitne teachers hai": "Har branch me 3 ya usse zyada teachers hote hain jo students ko guide karte hain.",

"teachers support karte hai kya": "Yes, teachers are very supportive and help students in academics and motivation.",
"faculty support kaisa hai": "Faculty support is very good. Teachers help in studies and also motivate students.",

"teachers help karte hai kya": "Yes, teachers always help students and guide them properly.",

"extra teachers hai kya": "Yes, college has sports teachers, English teachers, yoga instructors and other specialized staff.",
"special teachers kaun kaun hai": "College me sports teacher, English teacher, yoga instructor aur other specialized teachers available hain.",

"overall development hota hai kya": "Yes, teachers support overall development including academics, sports and personality growth.",

"faculty details": "College faculty includes experienced professors, supportive teachers and specialized staff for overall student development.",
# transportation
"transportation": "College has good transportation facilities. Buses and auto-rickshaws are easily available and travel is convenient.",
"transport facility": "Transportation is smooth. Students can easily get buses and autos near the college.",

"transportation kaisa hai": "College ka transport system achha hai. Buses aur autos easily available hote hain.",
"college ka transport kaisa hai": "Travel easy aur convenient hai. Buses aur auto-rickshaw mil jate hain.",

"bus facility hai kya": "Yes, buses are easily available near the college.",
"college me bus milti hai kya": "Yes, college ke bahar buses easily mil jati hain.",

"auto milta hai kya": "Yes, auto-rickshaws are easily available for students.",
"auto rickshaw available hai kya": "Yes, autos easily mil jate hain travel ke liye.",

"railway station kitna dur hai": "Railway station is about 1 km away from the college.",
"train se aa sakte hai kya": "Yes, railway station sirf 1 km dur hai, students easily train se aa sakte hain.",

"college jana easy hai kya": "Yes, transportation is smooth and hassle-free for students.",
"travel me problem hoti hai kya": "No, travel easy hai aur koi problem nahi hoti.",

"transport details": "College has good connectivity with buses, autos and nearby railway station (1 km).",
# environment
"college environment": "College environment is very convenient and student-friendly with many shops and facilities nearby.",
"environment kaisa hai": "College ka environment bahut achha aur student-friendly hai.",

"college ke aas paas kya hai": "College ke paas cyber cafes, shops aur tea stalls available hain.",
"campus ke bahar kya facilities hai": "Campus ke bahar cyber cafe, chai stall aur zaroori shops available hain.",

"cyber cafe hai kya": "Yes, cyber cafes are available near the college for online work and form filling.",
"online kaam kaha hota hai": "Students cyber cafe me jaakar online kaam aur form filling kar sakte hain.",

"tea stall hai kya": "Yes, tea stalls and small shops are available near the college for refreshment.",
"chai pani milta hai kya": "Yes, chai aur snacks ke liye stalls available hain.",

"college me cafe hai kya": "Yes, there is a Sudha Café inside the college campus for refreshments.",
"sudha cafe kya hai": "Sudha Café college ke andar hai jaha students refreshment le sakte hain.",

"students ke liye facilities hai kya": "Yes, all necessary items and facilities for students are easily available near the college.",
"zaroori saman milta hai kya": "Yes, college ke paas sab zaroori saman easily mil jata hai.",

"environment details": "College environment is friendly with nearby shops, cyber cafes and Sudha Café inside campus.",
# enrollment
"enrollment process": "Students have to complete enrollment every semester with proper guidance from teachers.",
"enrollment kya hota hai": "Enrollment har semester me hota hai jisme students apni registration complete karte hain.",

"enrollment kab hota hai": "Teachers inform students about enrollment dates and process on time.",
"enrollment kaise hota hai": "Teachers guide students step by step for completing enrollment and forms.",

"form kaise fill kare": "Teachers provide proper guidance for filling all academic forms.",
"exam form kaise bhare": "Students are guided by teachers on how and when to fill exam forms.",

"exam schedule kaise pata chalega": "Teachers inform exam schedule clearly and on time.",
"exam kab hota hai": "Exam dates and schedule are shared by teachers and through notices.",

"college information kaise milta hai": "All important information is shared clearly by teachers and through WhatsApp groups.",
"updates kaha milte hai": "Students get updates in WhatsApp groups with PDFs and instructions.",

"whatsapp group hai kya": "Yes, important updates and notices are shared in WhatsApp groups.",
"notice kaise milta hai": "Notices and instructions are shared via teachers and WhatsApp groups.",

"students ko help milti hai kya": "Yes, teachers provide full guidance so students do not face confusion.",
"academic help milta hai kya": "Yes, students get proper support and guidance for all academic processes.",

"confusion hota hai kya": "No, all information is shared clearly so students do not face confusion.",

"enrollment details": "Enrollment is done every semester with full teacher guidance and updates via WhatsApp groups.",
# events
"events": "College organizes various events and festivals with enthusiasm like Saraswati Puja, sports and national celebrations.",
"college events": "College celebrates Saraswati Puja, Umang sports competition and national festivals.",

"events kaun kaun se hote hai": "College me Saraswati Puja, sports competition aur national festivals celebrate hote hain.",
"college me kya kya events hote hai": "College me religious, cultural aur sports events organize kiye jate hain.",

"saraswati puja hota hai kya": "Yes, Saraswati Puja is celebrated with devotion and cultural activities.",
"saraswati puja kaise hota hai": "Students Saraswati Puja me bhag lete hain with devotion and cultural spirit.",

"sports hota hai kya": "Yes, Umang Sports Competition is organized for students.",
"umang kya hai": "Umang ek sports competition hai jisme students apni talent dikhate hain.",

"independence day celebrate hota hai kya": "Yes, Independence Day is celebrated with pride and patriotism.",
"republic day celebrate hota hai kya": "Yes, Republic Day is celebrated with pride and patriotism.",

"teachers day hota hai kya": "Yes, Teacher’s Day is celebrated to show respect and gratitude to teachers.",
"teachers day kaise celebrate hota hai": "Students celebrate Teacher’s Day to thank and respect teachers.",

"events ka benefit kya hai": "Events create a lively environment and help in overall development of students.",
"events important kyu hai": "Events help students in personality development beyond academics.",

"college environment lively hai kya": "Yes, events and celebrations make the college environment lively and enjoyable.",

"events details": "College organizes Saraswati Puja, Umang sports, national festivals and Teacher’s Day for student development.",
# assignments
"assignments": "Students have to prepare subject-wise assignments and lab files every semester.",
"assignment kya hota hai": "Assignments har subject ke according banaye jate hain jo teachers guide karte hain.",

"assignment banana padta hai kya": "Yes, har semester me assignments aur lab files banana compulsory hota hai.",
"lab file banana padta hai kya": "Yes, lab files banana zaruri hota hai har semester me.",

"assignment ka benefit kya hai": "Assignments help students revise and understand concepts better.",
"assignment kyu important hai": "Assignments se concepts clear hote hain aur revision hota hai.",

"assignment check hota hai kya": "Yes, assignments and lab files are checked and signed by subject teachers.",
"assignment kab check hota hai": "Final exam se pehle assignments aur lab files check hote hain.",

"assignment submit karna hota hai kya": "Yes, assignments must be submitted on time for marks.",
"assignment marks milta hai kya": "Yes, marks are given based on submission and quality of work.",

"internal marks kaise milta hai": "Internal marks depend on assignments, lab work and attendance.",
"evaluation kaise hota hai": "Evaluation is based on assignments, lab files, attendance and performance.",

"attendance important hai kya": "Yes, attendance is very important for internal assessment.",
"attendance ka kya role hai": "Attendance plays a major role in evaluation and internal marks.",

"assignment details": "Students must complete assignments and lab files every semester which are checked before exams and used for evaluation.",

# anti ragging
"anti ragging policy": "Ragging is strictly prohibited in the college. Strict action is taken against anyone involved.",
"ragging allowed hai kya": "No, ragging is strictly prohibited in the college.",

"ragging hota hai kya": "No, the college follows a zero-tolerance policy against ragging.",
"ragging rules kya hai": "College me ragging ke against strict rules hain aur zero tolerance policy follow hoti hai.",

"ragging kare to kya hoga": "Strict action will be taken against students involved in ragging.",
"ragging karne par kya punishment hai": "Ragging karne par strict disciplinary action liya jata hai.",

"ragging report kaise kare": "Students can report ragging to teachers or college authorities immediately.",
"ragging complaint kaha kare": "Students teachers ya authorities ko bina hesitation report kar sakte hain.",

"college safe hai kya": "Yes, college provides a safe, respectful and friendly environment.",
"environment safe hai kya": "Yes, college ensures a safe and friendly environment for all students.",

"students ko safety milti hai kya": "Yes, students are provided with a safe and respectful environment.",
"ragging se protection milta hai kya": "Yes, college ensures full protection and strict action against ragging.",

"ragging details": "Ragging is strictly banned and students can report any issue to teachers or authorities.",
# campus environment
"campus environment": "College campus is beautiful, green and well-maintained with a peaceful environment.",
"campus kaisa hai": "College campus bahut sundar, green aur well-maintained hai.",

"campus me greenery hai kya": "Yes, campus me greenery aur flowers hai jo environment ko pleasant banate hain.",
"campus clean hai kya": "Yes, campus clean aur well-maintained hai.",

"campus safe hai kya": "Yes, campus me CCTV cameras lage hue hain for safety and security.",
"security kaise hai": "Campus me CCTV cameras ke through proper security aur monitoring hoti hai.",

"college ka environment kaisa hai": "Environment peaceful, clean aur student-friendly hai.",
"environment safe hai kya": "Yes, environment safe aur disciplined hai.",

"campus details": "Campus is green, clean and secured with CCTV cameras providing a peaceful environment.",
# classroom
"classroom facilities": "Classrooms are well-equipped with smart boards, CCTV, fans and proper lighting.",
"classroom kaisa hai": "Classrooms modern aur well-equipped hain with smart boards aur facilities.",

"classroom me kya facilities hai": "Classroom me smart board, CCTV, fans aur proper lighting available hai.",
"classroom smart hai kya": "Yes, classrooms me smart boards available hain.",

"cctv classroom me hai kya": "Yes, classrooms me CCTV cameras installed hain.",
"har class me camera h kya": "Yes, classrooms me bhi CCTV cameras hain.",

"fan light proper hai kya": "Yes, classrooms me proper fans aur lighting hai.",
"charging point hai kya": "Yes, students ke liye charging points available hain.",

"classroom comfortable hai kya": "Yes, classrooms comfortable aur learning ke liye suitable hain.",
"learning environment kaisa hai": "Classrooms smart aur comfortable learning environment provide karte hain.",

"classroom details": "Classrooms are modern with smart boards, CCTV, lighting, fans and charging points.",
# sports
"sports activities": "College organizes annual sports competitions where students participate in various games and activities.",
"sports hota hai kya": "Yes, college me har saal sports competition hota hai jisme students participate karte hain.",

"college me sports hota hai kya": "Yes, annual sports events organize kiye jate hain for all branches.",
"sports competition kab hota hai": "Sports competition har saal organize kiya jata hai.",

"kaun kaun se sports hote hai": "College me cricket, volleyball, kabaddi, chess, carrom, long jump, high jump aur other activities hote hain.",
"sports me kya kya hota hai": "Cricket, volleyball, kabaddi, chess, carrom, singing, painting aur athletics activities hoti hain.",

"students participate karte hai kya": "Yes, students apni teams bana kar sports me participate karte hain.",
"team bana sakte hai kya": "Yes, students apni team bana kar sports events me participate kar sakte hain.",

"sports ka benefit kya hai": "Sports se teamwork, talent aur personality development hota hai.",
"sports important kyu hai": "Sports se physical aur mental development hota hai.",

"winners ko kya milta hai": "Winners ko higher level competitions me participate karne ka chance milta hai.",
"jeetne par kya hota hai": "Winners ko next level competitions ke liye select kiya jata hai.",

"teachers support karte hai kya": "Yes, teachers students ko motivate aur support karte hain sports me participate karne ke liye.",
"teachers encourage karte hai kya": "Yes, teachers encourage students for sports and extracurricular activities.",

"sports details": "College organizes annual sports with many games and activities for overall student development.",
# study facilities
"study facilities": "College provides excellent study facilities including library and computer labs.",
"study facilities kaisi hai": "College me study facilities bahut achhi hain with library aur labs.",

"library hai kya": "Yes, college me well-equipped library available hai with many books.",
"library kaisa hai": "Library me different subjects ki books available hain aur students waha study kar sakte hain.",

"library card milta hai kya": "Yes, students ko library use karne ke liye library card diya jata hai.",
"book issue kaise hoti hai": "Library card ke through students books issue kar sakte hain.",

"library me padh sakte hai kya": "Yes, students library me baith kar peaceful environment me padh sakte hain.",
"library environment kaisa hai": "Library ka environment quiet aur focused study ke liye suitable hai.",

"computer lab hai kya": "Yes, college me computer labs available hain for study.",
"matlab lab hai kya": "Yes, MATLAB lab bhi available hai for students.",

"lab me kya kar sakte hai": "Students labs me computer use karke study kar sakte hain aur practice kar sakte hain.",
"online study kar sakte hai kya": "Yes, students educational videos dekh sakte hain aur digital learning resources use kar sakte hain.",

"digital learning facilities hai kya": "Yes, college provides digital learning through computer labs and internet resources.",
"technology se study hota hai kya": "Yes, students technology aur computers ke through study kar sakte hain.",

"study ka environment kaisa hai": "College me study ke liye peaceful aur supportive environment hai.",

"study details": "College provides library, computer labs and digital learning resources for better study experience.",
# laboratory
"laboratory facilities": "College has many labs like Chemistry, Physics, Computer, COE, MATLAB and KYP for practical learning.",
"lab facilities kaisi hai": "College me lab facilities bahut achhi hain with different specialized labs.",

"college me kaun kaun se lab hai": "College me Chemistry, Physics, Computer, COE, MATLAB aur KYP labs available hain.",
"labs kaun kaun se hai": "Different labs jaise Chemistry, Physics, Computer aur specialized labs available hain.",

"computer lab hai kya": "Yes, computer lab available hai for students.",
"matlab lab hai kya": "Yes, MATLAB lab bhi available hai.",

"coe lab kya hai": "COE (Center of Excellence) lab ek specialized lab hai jaha advanced learning hoti hai.",
"kyp lab kya hai": "KYP lab skill development ke liye use hota hai jaha students practical training lete hain.",

"practical hota hai kya": "Yes, students labs me experiments perform karte hain.",
"lab me kya karte hai": "Students experiments karte hain aur practical knowledge gain karte hain.",

"practical knowledge milta hai kya": "Yes, labs se students ko practical knowledge aur hands-on experience milta hai.",
"hands on experience milta hai kya": "Yes, labs me students ko real practical experience milta hai.",

"lab teachers guide karte hai kya": "Yes, teachers proper guidance dete hain during practical sessions.",
"lab me help milti hai kya": "Yes, teachers students ko experiments me guide karte hain.",

"lab ka benefit kya hai": "Labs help students understand theory better and develop practical skills.",
"labs important kyu hai": "Labs se practical skills develop hoti hain jo future ke liye important hain.",

"lab details": "College provides various labs for practical learning and skill development in each department.",
# workshop
"workshop facilities": "College organizes workshops every semester to improve students' practical knowledge and skills.",
"workshop hota hai kya": "Yes, college me har semester workshops organize hote hain.",

"college me workshop hota hai kya": "Yes, workshops regularly conduct kiye jate hain for skill development.",
"workshop kab hota hai": "Workshops har semester me organize kiye jate hain.",

"workshop ka benefit kya hai": "Workshops se students ko practical knowledge aur technical skills milti hain.",
"workshop kyu important hai": "Workshops se real world skills aur understanding improve hoti hai.",

"cs ke liye workshop hota hai kya": "Yes, Computer Science students ke liye web development aur technical workshops hote hain.",
"web development workshop hota hai kya": "Yes, web development aur other technical workshops conduct kiye jate hain.",

"all branch ke liye workshop hota hai kya": "Yes, kuch workshops sabhi branches ke students ke liye bhi hote hain.",
"common workshop hota hai kya": "Yes, interdisciplinary workshops bhi conduct kiye jate hain.",

"company workshop karati hai kya": "Yes, companies jaise Infosys aur Infochord workshops conduct karte hain.",
"infosys workshop hota hai kya": "Yes, Infosys jaise reputed companies workshops conduct karte hain.",

"industry exposure milta hai kya": "Yes, workshops se students ko industry exposure milta hai.",
"practical experience milta hai kya": "Yes, workshops me hands-on experience milta hai.",

"workshop details": "College organizes workshops every semester including technical and company-based sessions for students.",
# industrial visit
"industrial visit": "Final year students are taken to industrial company visits to understand real working environment.",
"industrial visit hota hai kya": "Yes, final year students ko industrial visits karayi jati hai.",

"company visit hota hai kya": "Yes, students ko companies me visit karaya jata hai to see how work is done.",
"industry visit hota hai kya": "Yes, students visit industries to learn about real work environment.",

"industrial visit kab hota hai": "Industrial visits mostly final year students ke liye organize ki jati hain.",
"final year me visit hota hai kya": "Yes, final year students ke liye company visits hoti hain.",

"industrial visit ka benefit kya hai": "Students ko real working environment samajhne ka mauka milta hai.",
"company visit kyu hota hai": "Students ko practical exposure aur real industry ka experience milta hai.",

"waha kya dekhte hai": "Students dekhte hain ki company me kaam kaise hota hai aur environment kaisa hota hai.",
"industry me kya sikhte hai": "Students real work process aur company environment ke bare me sikhte hain.",

"real experience milta hai kya": "Yes, industrial visits se students ko real experience milta hai.",
"practical exposure milta hai kya": "Yes, students ko industry ka practical exposure milta hai.",

"industrial visit details": "Final year students visit industries to understand real work process and environment.",
# ------------------ ADMISSION ------------------

"how to take admission": "Admission is done through entrance exam and counseling process. Students need to qualify exam, choose college and branch, then complete document verification and fee payment.",
"admission kaise le": "Admission entrance exam aur counseling ke through hota hai. Exam qualify karna hota hai, fir college aur branch choose karke documents verify aur fees submit karni hoti hai.",

"admission process": "Admission is conducted through entrance exam followed by counseling and document verification.",
"admission process kya hai": "Admission process me entrance exam, counseling, document verification aur fee payment hota hai.",

"eligibility for admission": "Students must pass 10th from a recognized board with required minimum marks.",
"admission eligibility kya hai": "Student ko 10th pass hona chahiye kisi recognized board se required marks ke sath.",

"can i get direct admission": "No, direct admission is usually not available. Admission is done through entrance exam and counseling.",
"direct admission milta hai kya": "Nahi, direct admission available nahi hota. Admission entrance exam ke through hota hai.",

"documents required for admission": "Required documents include 10th marksheet, admit card, Aadhaar card, photos, caste certificate, income certificate, domicile and migration certificate if required.",
"admission ke liye documents kya chahiye": "10th marksheet, admit card , Aadhaar card, photo, caste, income, domicile aur migration certificate (agar required ho) chahiye if you are 12th pass then 12th ka v marksheet ,rank list ,counselling form .",
"original documents required": "Yes, original documents are required for verification during admission along with photocopies.",
"kya original documents dena padta hai": "Haan, admission ke time verification ke liye original documents dena padta hai aur photocopy bhi submit hoti hai.",

"migration certificate required": "Migration certificate is required if you are from a different board.",
"migration certificate zaruri hai kya": "Agar aap different board se ho to migration certificate zaruri hota hai.",

"what if no migration certificate": "It is advised to arrange migration certificate before admission if required.",
"migration certificate nahi hai to kya kare": "Agar migration certificate required hai to admission se pehle arrange kar lena chahiye but clc ho to admission ho jayega.",

"documents details": "Students must submit required documents like marksheet, Aadhaar, certificates and photos during admission process.",
# ------------------ MARKS & EVALUATION ------------------

"marks pattern": "Final result is based on assignments, lab work, attendance, class participation, internal tests, viva and external exams.",
"evaluation system": "Students are evaluated based on internal assessment and external examination along with assignments and practical work.",

"marks kaise milte hai": "Marks assignments, lab work, attendance, internal tests, viva aur external exam ke basis par milte hain.",
"college me evaluation kaise hota hai": "Evaluation internal aur external dono basis par hota hai, jisme assignments aur practical work bhi include hota hai.",

"external exam marks": "External exam is usually of 70 marks.",
"external exam kitne marks ka hota hai": "External exam generally 70 marks ka hota hai.",

"internal marks": "Internal assessment is around 20 to 30 marks given by subject teachers.",
"internal marks kitne hote hai": "Internal marks 20 se 30 ke beech hote hain jo teachers dete hain.",

"internal marks kaise milta hai": "Internal marks assignments, attendance, class participation aur tests ke basis par milta hai.",
"internal assessment kaise hota hai": "Internal assessment teachers dete hain based on overall performance.",

"exam pattern": "Exam includes objective, short answer and long answer questions, The external examination generally follows this pattern.",
"paper pattern kya hai": "Paper me objective, short answer aur long answer questions hote hain, The external examination generally follows this pattern.",

"objective questions kitne hote hai": "There are 10 objective questions of 2 marks each.",
"objective question pattern": "10 objective questions hote hain, har ek 2 marks ka hota hai.",

"short answer questions": "There are 5 short answer questions of 4 marks each.",
"short questions kitne hote hai": "5 short answer questions hote hain, har ek 4 marks ka hota hai.",

"long answer questions": "There are 5 long answer questions of 6 marks each.",
"long questions kitne hote hai": "5 long answer questions hote hain, har ek 6 marks ka hota hai.",

"viva hota hai kya": "Yes, viva is also part of internal evaluation.",
"viva marks milta hai kya": "Yes, viva ke marks bhi internal assessment me include hote hain.",

"attendance important hai kya": "Yes, attendance plays an important role in internal marks.",
"attendance ka role kya hai": "Attendance internal marks aur evaluation me important role play karta hai.",

"practical marks kaise milta hai": "Practical marks lab work aur performance ke basis par milta hai.",
"lab marks kaise milta hai": "Lab marks practical performance aur file submission ke basis par milta hai.",

"overall evaluation kaise hota hai": "Overall evaluation exams ke sath-sath assignments, practical work aur attendance par depend karta hai.",

"marks details": "Students are evaluated through external exams (70 marks) and internal assessment (20-30 marks) including assignments, viva and attendance.",
# ------------------ FINAL YEAR PROJECT ------------------

"final year project": "In final year, students have to make a working project based on their branch to gain practical knowledge.",
"final year me kya hota hai": "Final year me students ko apne branch ke according ek working project banana hota hai.",

"project banana padta hai kya": "Yes, final year me project banana compulsory hota hai.",
"final year project compulsory hai kya": "Haan, final year me project banana zaruri hota hai.",

"project ka benefit kya hai": "Project se students practical knowledge aur real-world experience gain karte hain.",
"project kyu important hai": "Project se students apna knowledge practically apply karte hain aur experience milta hai.",

# seminars
"seminar hota hai kya": "Yes, college seminars aur webinars organize karta hai jaha students presentation dete hain.",
"seminar kya hota hai": "Seminar me students ek topic choose karke PPT banate hain aur presentation dete hain.",

"ppt banana padta hai kya": "Yes, seminar ke liye students ko PPT banana hota hai.",
"presentation dena padta hai kya": "Yes, students ko seminar me presentation dena hota hai.",

"seminar ka benefit kya hai": "Seminar se presentation skill, confidence aur technical knowledge improve hota hai.",
"seminar kyu important hai": "Seminar se confidence aur communication skills improve hoti hain.",

"webinar hota hai kya": "Yes, college webinars bhi organize karta hai for student learning.",
"seminar details": "Final year me project aur seminars hote hain jo students ke skills aur confidence improve karte hain.",
# ------------------ MESS FACILITIES ------------------

"mess facility": "College provides mess facility for students with veg and non-veg food options.",
"mess hai kya": "Yes, college me mess facility available hai students ke liye.",

"mess kaisa hai": "Mess me different food options available hote hain aur weekly menu follow hota hai.",
"mess ka system kya hai": "Mess me 7-day weekly menu hota hai aur students apni choice se select kar sakte hain.",

"boys hostel mess": "Mess boys hostel me available hai jaha students directly khana khate hain.",
"boys ke liye mess kaha hai": "Boys hostel me mess facility available hai.",

"girls hostel mess": "Girls hostel ke liye food tiffin me pack karke din me 3 baar deliver kiya jata hai.",
"girls ke liye mess kaise hota hai": "Girls hostel me food tiffin ke through 3 times daily deliver hota hai.",

"khana kitni baar milta hai": "Food din me 3 baar provide kiya jata hai students ko.",
"meal timing kya hai": "Students ko daily 3 time meals milta hai.",

"veg non veg available hai kya": "Yes, both veg and non-veg food options are available.",
"veg khana milta hai kya": "Yes, veg food available hai mess me.",
"non veg milta hai kya": "Yes, non-veg food bhi available hai.",

"mess option kitne hai": "Students ke liye 2 different mess options available hain.",
"mess choose kar sakte hai kya": "Yes, students apni preference ke according mess choose kar sakte hain.",

"mess details": "College provides mess with weekly menu, veg/non-veg options and separate system for boys and girls.",
# ------------------ CARRY / BACKLOG ------------------

"backlog kya hota hai": "Agar student kisi subject me fail ho jata hai to use backlog ya carry kehte hain.",
"carry kya hota hai": "Carry ka matlab hai kisi subject me fail hona jise baad me clear karna hota hai.",

"backlog kaise clear kare": "Students backlog ko next semester ke exam me clear kar sakte hain.",
"carry kaise clear hota hai": "Carry subject ko upcoming semester exams me pass karke clear kiya jata hai.",

"kitne chance milte hai backlog clear karne ke liye": "Students ko backlog clear karne ke liye maximum 6 chances milte hain.",
"backlog attempts kitne hote hai": "Backlog clear karne ke liye 6 attempts tak milte hain.",

"backlog clear nahi hua to kya hoga": "Agar student 6 attempts me backlog clear nahi kar pata hai to use NFT certificate milta hai.",
"fail hone par kya hota hai": "Agar backlog clear nahi hota to student ko NFT (Not Fit for Technical) mil sakta hai.",

"nft kya hota hai": "NFT ka matlab Not Fit for Technical hota hai jo further admission me problem create kar sakta hai.",
"nft ka full form kya hai": "NFT ka full form Not Fit for Technical hai.",

"nft milne par kya hota hai": "NFT milne par dusre college me admission lene me problem hoti hai.",
"nft ke baad admission hota hai kya": "NFT ke baad admission lena difficult ho jata hai.",

"year back kab lagta hai": "Agar student 3 ya usse zyada subjects me fail ho jata hai aur unhe clear nahi karta to year back lag sakta hai.",
"year back kya hota hai": "Year back ka matlab hai student ko same year repeat karna padta hai.",

"year back kyu lagta hai": "Jab student multiple subjects me fail ho jata hai aur unhe clear nahi karta tab year back lagta hai.",
"year back me kya hota hai": "Year back me student next semester me nahi ja pata aur same year dobara padhna padta hai.",

"next semester me ja sakte hai kya backlog ke sath": "Agar backlog kam hai to student next semester me ja sakta hai, lekin zyada subjects me fail hone par year back lag sakta hai.",

"backlog system details": "Students can clear failed subjects in next exams with up to 6 attempts, otherwise NFT may be given and year back can happen.",
# ------------------ GYM FACILITIES ------------------

"gym facility": "College provides gym facility in boys hostel for students fitness and workout.",
"gym hai kya": "Yes, college me gym facility available hai boys hostel me.",

"gym kaha hai": "Gym boys hostel me located hai jaha students workout kar sakte hain.",
"college me gym kaha hai": "Gym facility boys hostel me available hai.",

"gym kaisa hai": "Gym me different exercise machines available hain jisse students workout kar sakte hain.",
"gym me kya facilities hai": "Gym me various machines aur equipment available hain for exercise.",

"workout kar sakte hai kya": "Yes, students gym me jaakar regular workout kar sakte hain.",
"fitness facility hai kya": "Yes, gym facility available hai fitness maintain karne ke liye.",

"girls hostel me gym hai kya": "Currently girls hostel me gym facility available nahi hai.",
"girls ke liye gym hai kya": "Nahi, abhi girls hostel me gym facility available nahi hai.",

"gym use kaise kare": "Students boys hostel me jaakar gym use kar sakte hain as per rules.",
"gym details": "Gym facility boys hostel me available hai with machines for workout, but girls hostel me abhi available nahi hai.",
# ------------------ HOSTEL FACILITIES ------------------

"hostel facility": "College provides separate hostel facilities for boys and girls with safe and comfortable environment.",
"hostel hai kya": "Yes, college me boys aur girls ke liye separate hostel available hai.",

"hostel kaisa hai": "Hostel safe, comfortable aur student-friendly hai jaha basic facilities available hain.",
"college hostel kaisa hai": "Hostel me safe aur convenient environment milta hai students ko.",

"boys hostel hai kya": "Yes, boys ke liye separate hostel available hai.",
"girls hostel hai kya": "Yes, girls ke liye bhi separate hostel available hai.",

"hostel room me kya hota hai": "Hostel rooms me almirah, study table, bed aur charging points available hote hain.",
"room facilities kya hai": "Room me basic facilities jaise bed, table, almirah aur charging point hota hai.",

"kitne student ek room me rehte hai": "Minimum 2 students ek room me stay karte hain.",
"room sharing kaise hota hai": "Ek room me generally 2 ya usse zyada students rehte hain.",

"hostel safe hai kya": "Yes, hostel safe aur secure environment provide karta hai students ke liye.",
"hostel security kaisi hai": "Hostel me proper safety aur security maintain ki jati hai.",

"hostel me rehna comfortable hai kya": "Yes, hostel comfortable aur study-friendly environment provide karta hai.",
"hostel environment kaisa hai": "Hostel me friendly aur cooperative environment hota hai jaha students aaram se reh sakte hain.",

"hostel details": "College provides separate boys and girls hostel with basic facilities and safe environment for students.",
# ------------------ WIFI FACILITY ------------------

"wifi facility": "WiFi is available in the college campus and hostel for students.",
"wifi hai kya": "Yes, college aur hostel dono me WiFi facility available hai.",

"college me wifi hai kya": "Yes, college campus me WiFi available hai students ke liye.",
"hostel me wifi hai kya": "Yes, hostel me bhi WiFi available hai.",

"internet facility hai kya": "Yes, students ke liye internet aur WiFi facility available hai.",
"wifi free hai kya": "College me students ke liye WiFi facility available hoti hai for study purposes.",

"wifi details": "College aur hostel dono jagah WiFi available hai jisse students online study kar sakte hain.",


# ------------------ EXTRA COURSES ------------------

"extra courses": "College provides extra courses like KYP, Cisco and other online courses with certificates.",
"extra course kya hai": "College me KYP, Cisco aur kai online courses available hain jinke certificates milte hain.",

"kyp kya hai": "KYP ek skill development course hai jisme students ko practical knowledge diya jata hai.",
"cisco course kya hai": "Cisco course ek technical course hai jo networking skills develop karta hai.",

"extra course milta hai kya": "Yes, students ke liye additional courses available hote hain skill development ke liye.",
"online courses milte hai kya": "Yes, college me online courses available hain jinke certificate milte hain.",

"certificate milta hai kya": "Yes, extra courses aur workshops ke completion par certificate milta hai.",
"course certificate milta hai kya": "Haan, course complete karne par certificate diya jata hai.",

"workshop certificate milta hai kya": "Yes, workshops attend karne par bhi certificate milta hai.",
"workshop me certificate milta hai kya": "Haan, workshop ke baad certificate diya jata hai.",

"skill development hota hai kya": "Yes, extra courses aur workshops se students ki skills develop hoti hain.",
"extra course ka benefit kya hai": "Extra courses se technical knowledge aur skills improve hoti hain.",

"courses details": "College provides KYP, Cisco and other online courses with certificates for skill development.",
# ------------------ COLLEGE TIMING ------------------

"college timing": "College runs from 9 AM to 5 PM with classes and labs throughout the day.",
"college ka time kya hai": "College ka time 9 baje se 5 baje tak hota hai.",

"class timing": "Classes start from 10 AM and continue till 5 PM with a lunch break in between.",
"class kab hoti hai": "Classes 10 baje start hoti hain aur 5 baje tak chalti hain.",

"lunch time": "Lunch break is at 1 PM and classes resume again from 2 PM.",
"lunch kab hota hai": "Lunch 1 baje hota hai aur 2 baje se fir class start hoti hai.",

"break timing kya hai": "College me 1 baje lunch break hota hai.",
"lunch break kitne baje hota hai": "Lunch break 1 PM par hota hai.",

"2 baje ke baad kya hota hai": "2 baje ke baad fir se classes aur labs start ho jati hain.",
"lunch ke baad class hoti hai kya": "Haan, lunch ke baad 2 baje se classes aur labs hoti hain.",

"lab kab hota hai": "Labs usually lunch ke baad ya schedule ke according hoti hain.",
"practical kab hota hai": "Practical classes 2 baje ke baad ya timetable ke according hoti hain.",

"full day schedule": "College 9 AM se 5 PM tak hota hai jisme 1 baje lunch break aur 2 baje se fir classes aur labs hoti hain.",

"timing details": "College timing is 9 AM to 5 PM with lunch at 1 PM and classes again from 2 PM.",
# ------------------ MEDICAL FACILITY ------------------

"medical facility": "College provides basic medical facility with important medical equipment for students.",
"medical facility hai kya": "Yes, college me basic medical facility available hai students ke liye.",

"college me medical facility hai kya": "Haan, college me ek chhota medical service available hai jisme zaroori equipment hote hain.",
"health facility hai kya": "Yes, students ke liye basic health aur medical support available hai.",

"medical room hai kya": "Yes, college me medical room available hai jaha basic treatment diya jata hai.",
"first aid milta hai kya": "Haan, students ko basic first aid aur medical help mil jati hai.",

"medical equipment hai kya": "Yes, medical facility me important equipment available hote hain.",
"emergency me help milti hai kya": "Haan, emergency situation me basic medical help available hoti hai.",

"doctor available hai kya": "Basic medical support available hai, lekin detailed treatment ke liye hospital jana padta hai.",
"treatment milta hai kya": "College me basic treatment aur first aid milta hai.",

"medical details": "College provides a small medical facility with basic equipment and first aid support for students.",
# ------------------ COMPLAINT SYSTEM ------------------

"complaint system": "Students can submit complaints through a register or directly inform teachers.",
"complaint kaise kare": "Students complaint register me likh sakte hain ya directly teacher ko bata sakte hain.",

"complaint register kya hai": "Complaint register ek system hai jisme students apni problems likhte hain.",
"register me complaint kaise kare": "Agar room ya hostel me problem ho to student register me likh deta hai.",

"complaint kaha kare": "Students complaint register me ya directly teachers ko inform kar sakte hain.",
"problem ho to kya kare": "Student apni problem register me likh sakta hai ya teacher ko bata sakta hai.",

"complaint check hota hai kya": "Yes, complaint register ko daily check kiya jata hai.",
"complaint ka solution hota hai kya": "Haan, complaints ko daily check karke solve kiya jata hai.",

"room problem kaise solve hoti hai": "Room ki problem register me likhne ke baad daily check hoti hai aur solve ki jati hai.",
"hostel problem kaise solve hoti hai": "Hostel problems register ke through ya teacher ko batakar solve ki jati hain.",

"teacher ko complaint kar sakte hai kya": "Yes, students directly teachers ko bhi apni problem bata sakte hain.",
"direct complaint kar sakte hai kya": "Haan, student directly teacher ko complaint kar sakta hai.",

"complaint details": "Students can write their issues in a register which is checked daily or inform teachers directly for quick solution.",
# ------------------ SYLLABUS & SUBJECTS ------------------

"syllabus kaha milega": "Syllabus official website par ya college department se milta hai.",
"where can i get syllabus": "The syllabus is available on the official website or from the college department.",

"syllabus kaise milega": "Students syllabus website ya apne department se le sakte hain.",
"syllabus details": "Syllabus official website ya college ke through provide kiya jata hai.",

"first year subjects": "First year me sabhi students ke liye common subjects hote hain jaise Maths, Physics, Chemistry aur basic engineering subjects.",
"1st year me kya subjects hote hai": "1st year me sabke liye same subjects hote hain jaise Maths, Physics, Chemistry etc.",

"first year me kya padhte hai": "Students first year me common subjects padhte hain jo sab branches ke liye same hote hain.",
"common subjects kya hote hai": "Common subjects me Maths, Physics, Chemistry aur basic engineering topics include hote hain.",

"second year subjects": "2nd year se students apne branch ke core subjects padhna start karte hain.",
"2nd year me kya padhte hai": "2nd year se students apne core (branch-specific) subjects padhte hain.",

"core subjects kya hote hai": "Core subjects wo hote hain jo student ke branch se related hote hain.",
"branch subjects kab start hote hai": "Branch ke subjects 2nd year se start hote hain.",

"subjects details": "1st year me common subjects hote hain aur 2nd year se students apne core branch subjects padhte hain.",
# ------------------ OFFICIAL WEBSITE ------------------

"official website": "Students use SBTE Bihar website for enrollment, exam form, fee payment and result.",
"college website": "Official website is https://sbteonline.bihar.gov.in/login for all student activities.",

"form filling website": "Students use https://sbteonline.bihar.gov.in/login for form filling and other services.",
"website kya hai": "SBTE Bihar ki official website hai jaha students apna academic work karte hain.",

"enrollment kaha hota hai": "Enrollment SBTE Bihar website par hota hai.",
"exam form kaha bhare": "Exam form SBTE Bihar website par fill kiya jata hai.",

"fee payment kaha hota hai": "Fee payment online SBTE Bihar website par hota hai.",
"result kaha dekhe": "Students apna result SBTE Bihar website par dekh sakte hain.",

"website ka use kya hai": "Is website ka use enrollment, exam form fill karne, fee payment aur result check karne ke liye hota hai.",
"online work kaha hota hai": "Students ka sara online work SBTE Bihar website par hota hai.",

"login website": "Students login karte hain SBTE Bihar portal par for all academic activities.",
"sbte website": "SBTE Bihar website students ke liye official portal hai jaha sab academic process hota hai.",

"website details": "SBTE Bihar website is used for enrollment, exam forms, fee payment and result checking.",
# ------------------ LATERAL ENTRY ------------------

"lateral entry kya hai": "Lateral entry ka matlab hai 12th ke baad direct 2nd year me admission lena.",
"lateral entry kya hota hai": "Lateral entry me students ko direct 2nd year me admission milta hai.",

"lateral entry admission": "Students can take admission directly in 2nd year through lateral entry after 12th.",
"lateral entry kaise hota hai": "12th ke baad students lateral entry ke through direct 2nd year me admission le sakte hain.",

"direct second year admission": "Yes, lateral entry me students ko direct 2nd year me admission milta hai.",
"2nd year me direct admission hota hai kya": "Haan, lateral entry ke through direct 2nd year me admission hota hai.",

"lateral entry eligibility": "Students who have completed 12th can apply for lateral entry admission.",
"lateral entry ke liye eligibility kya hai": "12th pass students lateral entry ke liye apply kar sakte hain.",

"lateral entry seats": "Each branch has limited seats for lateral entry students.",
"lateral entry me seat kitni hoti hai": "Har branch me lateral entry ke liye limited seats hoti hain.",

"lateral entry me extra subject hota hai kya": "Yes, lateral entry students ko extra subjects padhna padta hai.",
"lateral entry me kya padhna padta hai": "Students ko extra subjects jaise mechanics aur drawing padhna padta hai.",

"extra subjects kya hote hai": "Extra subjects me mechanics, engineering drawing jaise topics include hote hain.",
"lateral entry me extra subject kyu hota hai": "Extra subjects isliye hote hain taaki students first year ka basic cover kar saken.",

"core subject bhi padhna hota hai kya": "Yes, students ko apne core subjects bhi padhna hota hai.",
"lateral entry me kya kya padhna padta hai": "Students ko extra subjects ke sath core branch subjects bhi padhna hota hai.",

"lateral entry exam hota hai kya": "Yes, extra aur core subjects dono ka exam hota hai.",
"lateral entry me exam kaise hota hai": "Students ko sabhi subjects ka exam dena padta hai.",

"lateral entry details": "Lateral entry allows students to join 2nd year directly after 12th with limited seats and extra subjects like mechanics and drawing.",
}
import random
app = Flask(__name__)

# ------------------ SIMPLE NATURAL ------------------
def make_natural(reply):
    return reply


# ------------------ NLP MATCHING ------------------
questions = list(qa_pairs.keys())

def normalize_text(text):
    return text.lower()

normalized_questions = [normalize_text(q) for q in questions]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(normalized_questions)


def get_best_match(user_input):
    try:
        user_input = normalize_text(user_input)

        user_vector = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vector, tfidf_matrix)

        index = similarity.argmax()
        score = similarity[0][index]

        print("User:", user_input)
        print("Best match:", questions[index])
        print("Score:", score)

        if score > 0.2:
            reply = qa_pairs[questions[index]]
        else:
            reply = "samajh nahi aaya"

        return make_natural(reply)

    except Exception as e:
        return f"ERROR: {str(e)}"


# ------------------ AI RESPONSE ------------------
def get_ai_response(user_message):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer in simple Hinglish as a college chatbot"},
                {"role": "user", "content": user_message}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print("AI ERROR:", e)
        return "AI abhi kaam nahi kar raha 😅"


# ------------------ ROUTES ------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get-response", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_message = data.get("message", "")

    reply = get_best_match(user_message)

    if reply == "samajh nahi aaya":
        reply = get_ai_response(user_message)

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
