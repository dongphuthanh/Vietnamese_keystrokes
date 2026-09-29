"use strict";

// Define session questions in both English and Vietnamese
let session1Questions = {
  en: [
    "Recall and tell about any childhood memory that makes you happy (e.g. any favorite place or family trip?) (Use between 90-150 words)",
    "List at least 3 benefits of vaccination (like HPV, Covid19, Hepatitis B...) for individuals and the public. (Use between 90-150 words)",
    "Your friend often feels tired and stressed during exam periods. Based on what you’ve learned about healthy habits, what advice would you give them? (Use between 90-150 words)",
    "How do you think playing video games affects the way Vietnamese students socialize and learn in school? (Use between 90-150 words)",
    "What's your opinion about gifted schools? Evaluate both pros and cons(Use between 90-150 words)",
    "Describe the activities that you would include in a book promoting campaign for Vietnamese youth age 20 to 30. (Use between 90-150 words)"
  ],
  vi: [
    "Nhớ lại và kể về bất kỳ kỷ niệm thời thơ ấu nào mà bạn cảm thấy vui (ví dụ: bất kỳ địa điểm yêu thích hoặc chuyến đi gia đình nào?) (Dùng trong 90-150 từ)",
    "Nêu ít nhất 3 lợi ích của việc tiêm chủng vắc xin (như HPV, Covid19, cúm...) đối với cá nhân và cộng đồng. (Dùng trong 90-150 từ)",
    "Bạn của bạn thường cảm thấy mệt mỏi và căng thẳng trong thời gian thi cử. Dựa trên những gì bạn đã học về thói quen lành mạnh, bạn sẽ khuyên bạn ấy điều gì? (Dùng trong 90-150 từ)",
    "Bạn nghĩ việc chơi trò chơi điện tử ảnh hưởng như thế nào đến cách học tập và giao lưu của học sinh Việt Nam ở trường? (Dùng trong 90-150 từ)",
    "Quan điểm của bạn về trường chuyên là gì? Những mặt tốt và xấu có thể là gì? (Dùng trong 90-150 từ)",
    "Hãy mô tả các hoạt động mà bạn sẽ đưa vào Chiến dịch thúc đẩy việc đọc cho thanh thiếu niên Việt Nam từ 20 đến 30 tuổi. (Dùng trong 90-150 từ)"
  ]
};

let session2Questions = {
  en: [
    "What happened in the event of September 2, 1945 and what is its historical significance to the Vietnamese people? (Use between 90-150 words).",
    "Explain the main reasons why traffic congestion is a problem in big cities like Ho Chi Minh City or Hanoi. (Use between 90-150 words)",
    "Use your knowledge of stress and worry in youths, list some ways to help young Vietnamese people reduce stress and maintain mental health (Use between 90-150 words).",
    "Analyze the effects of early exposure to digital devices on children. (Use between 90-150 words).",
    "Should smartphones be banned in Vietnamese classrooms? Evaluate the pros and cons. (Use between 90-150 words)",
    "If you were tasked with creating a volunteer program to help protect the environment in your local area, how would you do it? What activities will your program have? (Use between 90-150 words)"
  ],
  vi: [
    "Sự kiện nào diễn ra vào ngày 2 tháng 9 năm 1945 và ý nghĩa lịch sử của nó đối với người dân Việt Nam là gì? (Dùng trong 90-150 từ).",
    "Giải thích những lý do chính khiến tình trạng tắc nghẽn giao thông trở thành vấn đề ở các thành phố lớn như Thành phố Hồ Chí Minh hoặc Hà Nội. (Dùng trong 90-150 từ)",
    "Sử dụng kiến thức của mình về chứng căng thẳng và lo âu ở vị thành niên, liệt kê một số cách giúp thanh thiếu niên Việt Nam giảm căng thẳng và duy trì sức khỏe tinh thần (Dùng trong 90-150 từ).",
    "Phân tích tác động của việc tiếp xúc sớm với các thiết bị điện tử đối với trẻ em. (Dùng trong 90-150 từ).",
    "Có nên cấm điện thoại thông minh trong các lớp học ở Việt Nam không? Đánh giá ưu và nhược điểm. (Dùng trong 90-150 từ)",
    "Nếu được giao nhiệm vụ tạo ra một chương trình tình nguyện để giúp bảo vệ môi trường tại địa phương của mình, bạn sẽ làm như thế nào? Chương trình của bạn sẽ có những hoạt động gì? (Dùng trong 90-150 từ)"
  ]
};

let session3Questions = {
  en: [
    "Name two Vietnamese historical figures along with some of their basic information and explain their significance (Use between 90-150 words).",
    "Why is the temperature in the city usually higher than in the countryside in summer? Explain based on knowledge of urban effect (Use between 90-150 words).",
    "In a situation that could potentially make you miss your deadline, what actions should you take? (Use between 90-150 words)",
    "Analyze the impact of social media trends on youth behavior in Vietnam. (Use between 90-150 words)",
    "Evaluate whether the use of AI tools like ChatGPT, Gemini or DeepSeek in studying helps or harms students' learning. (Use between 90-150 words)",
    "If you could develop an app to help solve a social problem in Vietnam, what would it be and how would your app work? (Use between 90-150 words)"
  ],
  vi: [
    "Nêu 2 nhân vật lịch sử của Việt Nam với vài thông tin đơn giản cùng tầm quan trọng của họ. (Dùng trong 90-150 từ).",
    "Tại sao trong mùa hè, nhiệt độ ở thành phố thường cao hơn ở nông thôn? Giải thích dựa trên kiến thức về hiệu ứng đô thị (Dùng trong 90-150 từ).",
    "Trong tình huống bạn có nguy cơ bị trễ deadline, bạn sẽ làm gì? (Dùng trong 90-150 từ)",
    "Phân tích ảnh hưởng của các xu hướng mạng đối với giới trẻ của Việt Nam. (Dùng trong 90-150 từ)",
    "Đánh giá xem việc sử dụng công cụ AI như ChatGPT, Gemini hay DeepSeek trong học tập có giúp ích hay gây hại cho việc học của học sinh. (Dùng trong 90-150 từ)",
    "Nếu bạn có thể phát triển một ứng dụng giúp giải quyết một vấn đề xã hội tại Việt Nam, đó sẽ là vấn đề gì và ứng dụng của bạn hoạt động như thế nào? (Dùng trong 90-150 từ)"
  ]
};

// Session instructions for each session in both English and Vietnamese
let sessionInstructions = {
  session1: {
    en: "Complete each question yourself in Vietnamese without using ChatGPT or any other external help.\nFor each question, write with less than 150 words.\nAnticipated duration: 30-40 minutes.",
    vi: "Tự trả lời câu hỏi bằng tiếng Việt và không sử dụng ChatGPT hay hỗ trợ từ ngoài.\nVới mỗi câu hỏi, trả lời trong không quá 150 từ.\nThời gian dự kiến: 30-40 phút."
  },
  session2: {
    en: "Copy the question (including the number of words required) and directly paste it into ChatGPT.\nCopy the generated response from ChatGPT and paste it into the corresponding input field indicated.\nParaphrase (in between 90 and 150 words) the generated response by going through sentence by sentence.\nKeep in mind, paraphrasing is rendering the same text in different words without losing the meaning of the text itself.\nRepeat the above steps for each question in this session.\nAnticipated duration: 30-40 minutes.",
    vi: "Copy câu hỏi (cả giới hạn số từ) và paste thẳng lên ChatGPT.\nCopy câu trả lời từ ChatGPT và Paste vào hộp văn bản dưới đây.\nDiễn giải lại (paraphrasing) câu trả lời của ChatGPT theo từng câu một.\nHãy nhớ rằng: diễn giải lại (paraphrasing) là viết lại câu bằng từ ngữ khác nhưng vẫn giữ nguyên ý nghĩa cũ của câu.\nLặp lại thao tác cho tất cả câu hỏi trong phần này.\nThời gian dự kiến: 30-40 phút."
  },
  session3: {
    en: "Copy the question (including the number of words required) and directly paste it into ChatGPT.\nCopy the generated response from ChatGPT and paste it into the corresponding input field indicated.\nRetype the response generated by ChatGPT.\nAnticipated duration: 30-40 minutes.",
    vi: "Copy câu hỏi (cả giới hạn số từ) và paste thẳng lên ChatGPT.\nCopy câu trả lời từ ChatGPT và Paste vào hộp văn bản dưới đây.\nChép nguyên câu trả lời của ChatGPT xuống hộp văn bản thứ hai.\nThời gian dự kiến: 30-40 phút."
  }
};

// Global variables to store inputs, keystrokes, and user information
let inputs = [];
let keystrokes = [];
let currentSession = 1;
let userInfo = {};
let totalQuestions = 0;
let language = 'en';

// ----------------------------------------------------------------------------
// 1) Improved logKeystroke: bail on inputs outside current session
// ----------------------------------------------------------------------------
function logKeystroke(event) {
  const inputIndex = inputs.findIndex(i => i.element === event.target);
  if (inputIndex < totalQuestions || inputIndex === -1) return;
  const questions =
    currentSession === 1 ? session1Questions[language]
    : currentSession === 2 ? session2Questions[language]
    : session3Questions[language];
  const perQ = currentSession === 1 ? 1 : 2;
  const qIndex = Math.floor((inputIndex - totalQuestions) / perQ);
  const question = questions[qIndex];
  
  // skip first "ChatGPT response" field in sessions 2 & 3
  if (currentSession !== 1 && ((inputIndex - totalQuestions) % 2) === 0) {
    return;
  }

  keystrokes.push({
    session: currentSession,
    question,
    key: event.key,
    code: event.code,
    event: event.type,
    timestamp: Date.now(),
    repeat: event.repeat
  });
}

// ----------------------------------------------------------------------------
// 2) Download only one session's keystrokes
// ----------------------------------------------------------------------------
function downloadSessionKeystrokes(sessionNum) {
  const sessionKeystrokes = keystrokes.filter(k => k.session === sessionNum);
  const blob = new Blob(
    [ JSON.stringify({ keystrokes: sessionKeystrokes }, null, 2) ],
    { type: 'application/json' }
  );
  createDownloadLink(blob, `keystrokes_session${sessionNum}.json`);

  // remove them so user won't re-download same data
  keystrokes = keystrokes.filter(k => k.session !== sessionNum);
}

// ----------------------------------------------------------------------------
// 3) Session flow with per-session download
// ----------------------------------------------------------------------------
function startSession() {
  hideAlert();
  document.getElementById('user-info-form').style.display = 'none';
  const container = document.getElementById('container');
  container.style.display = 'block';

  // Title & instructions
  document.querySelectorAll('h2.session-title').forEach(e => e.remove());
  const sub = document.createElement('h2');
  sub.className = 'center-text session-title';

  if (currentSession === 1) {
    sub.textContent = language === 'en' ? 'Bonafide Writing' : 'Viết chân chính';
    renderQuestions(container, session1Questions[language], false);
  } else if (currentSession === 2) {
    sub.textContent = language === 'en' ? 'Paraphrasing ChatGPT' : 'Diễn giải lại ChatGPT';
    renderQuestions(container, session2Questions[language], true);
  } else {
    sub.textContent = language === 'en' ? 'Retyping ChatGPT' : 'Chép lại ChatGPT';
    renderQuestions(container, session3Questions[language], true);
  }
  document.querySelector('h1').after(sub);

  const btn = document.createElement('button');
  btn.className = 'btn center-text';
  btn.textContent = currentSession < 3
    ? (language === 'en' ? 'Next Session' : 'Phần tiếp theo')
    : (language === 'en' ? 'Submit' : 'Nộp');
  container.appendChild(btn);

  btn.addEventListener('click', () => {
    if (currentSession === 1) {
      if (!checkAllAnswered(session1Questions[language], false)) {
        return showAlert(language==='en'
          ? 'Please answer all questions before proceeding.'
          : 'Hãy trả lời hết các câu hỏi trước khi tiếp tục.');
      }
      downloadSessionKeystrokes(1);
      totalQuestions += session1Questions[language].length;
      currentSession = 2;
      container.innerHTML = '';
      startSession();

    } else if (currentSession === 2) {
      if (!checkAllAnswered(session2Questions[language], true)) {
        return showAlert(language==='en'
          ? 'Please answer all questions in the current session.'
          : 'Hãy trả lời hết các câu hỏi của phần này.');
      }
      downloadSessionKeystrokes(2);
      totalQuestions += session2Questions[language].length * 2;
      currentSession = 3;
      container.innerHTML = '';
      startSession();

    } else {
      if (!checkAllAnswered(session3Questions[language], true)) {
        return showAlert(language==='en'
          ? 'Please answer all questions in the current session.'
          : 'Hãy trả lời hết các câu hỏi của phần này.');
      }
      downloadSessionKeystrokes(3);
      totalQuestions += session3Questions[language].length * 2;
      submitForm();
    }
  });
}

// ----------------------------------------------------------------------------
// 4) Render questions (single or dual-input mode)
// ----------------------------------------------------------------------------
function renderQuestions(container, questions, twoInputs = false) {
  // Instructions box
  const inst = document.createElement('div');
  inst.className = 'instruction-box';
  const p = document.createElement('p');
  p.textContent = language==='en' ? 'Steps to follow:' : 'Các bước thực hiện';
  inst.appendChild(p);
  const ol = document.createElement('ol');
  sessionInstructions[`session${currentSession}`][language]
    .split('\n').forEach(line => {
      const li = document.createElement('li');
      li.textContent = line.trim();
      ol.appendChild(li);
    });
  inst.appendChild(ol);
  container.appendChild(inst);

  // Each question
  questions.forEach((q, idx) => {
    const div = document.createElement('div');
    div.className = 'question';
    container.appendChild(div);

    const h2 = document.createElement('h2');
    h2.textContent = `${idx + 1}. ${q}`;
    div.appendChild(h2);

    if (twoInputs) {
      // ChatGPT response field
      const lbl1 = document.createElement('label');
      lbl1.textContent = language==='en'
        ? 'Copy Paste the generated response from ChatGPT:'
        : 'Copy Paste câu trả lời từ ChatGPT:';
      div.appendChild(lbl1);
      const ta1 = document.createElement('textarea');
      ta1.className = 'input';
      ta1.rows = 10;
      div.appendChild(ta1);
      inputs.push({ element: ta1, type: 'chatgpt-response' });
      const wc1 = document.createElement('div');
      wc1.className = 'word-count';
      wc1.textContent = language==='en' ? 'Word count: 0' : 'Số từ: 0';
      div.appendChild(wc1);
      ta1.addEventListener('input', () => updateWordCount(ta1, wc1));
      ta1.addEventListener('copy', e => e.stopPropagation());
      ta1.addEventListener('paste', e => e.stopPropagation());

      // User paraphrase / retype field
      const lbl2 = document.createElement('label');
      lbl2.textContent = currentSession===2
        ? (language==='en'
            ? 'Paraphrase the above response from ChatGPT (in 90-150 words):'
            : 'Diễn giải lại (paraphrasing) câu trả lời từ ChatGPT (Dùng trong 90-150 từ):')
        : (language==='en'
            ? 'Retype the generated response from ChatGPT:'
            : 'Đánh máy lại câu trả lời của ChatGPT:');
      div.appendChild(lbl2);
      const ta2 = document.createElement('textarea');
      ta2.className = 'input';
      ta2.rows = 10;
      div.appendChild(ta2);
      inputs.push({ element: ta2, type: 'user-response' });
      const wc2 = document.createElement('div');
      wc2.className = 'word-count';
      wc2.textContent = language==='en' ? 'Word count: 0' : 'Số từ: 0';
      div.appendChild(wc2);
      ta2.addEventListener('input', () => updateWordCount(ta2, wc2));
      ta2.addEventListener('copy', e => e.preventDefault());
      ta2.addEventListener('paste', e => e.preventDefault());

    } else {
      // Single direct-answer field
      const ta = document.createElement('textarea');
      ta.className = 'input';
      ta.rows = 10;
      div.appendChild(ta);
      inputs.push({ element: ta, type: 'user-response' });
      const wc = document.createElement('div');
      wc.className = 'word-count';
      wc.textContent = language==='en' ? 'Word count: 0' : 'Số từ: 0';
      div.appendChild(wc);
      ta.addEventListener('input', () => updateWordCount(ta, wc));
      if (currentSession !== 2) {
        ta.addEventListener('copy', e => e.preventDefault());
        ta.addEventListener('paste', e => e.preventDefault());
      }
    }
  });
}

// ----------------------------------------------------------------------------
// 5) Word count and answer-checking
// ----------------------------------------------------------------------------
function updateWordCount(textarea, displayDiv) {
  const count = textarea.value.trim().split(/\s+/).filter(w => w).length;
  displayDiv.textContent = language==='en'
    ? `Word count: ${count}`
    : `Số từ: ${count}`;
}

function checkAllAnswered(questions, twoInputs = false) {
  const needed = questions.length * (twoInputs ? 2 : 1);
  return inputs
    .slice(totalQuestions, totalQuestions + needed)
    .every(i => i.element.value.trim() !== '');
}

// ----------------------------------------------------------------------------
// 6) Download & submit
// ----------------------------------------------------------------------------
function createDownloadLink(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function submitForm() {
  // responses_data.json
  const s1 = session1Questions[language].length;
  const s2 = session2Questions[language].length * 2;
  const s3 = session3Questions[language].length * 2;

  const resp1 = session1Questions[language].map((q,i) => ({
    session: 1,
    question: q,
    wordCount: getWordCount(inputs[i].element.value),
    answer: inputs[i].element.value
  }));

  const resp2 = session2Questions[language].map((q,i) => ({
    session: 2,
    question: q,
    chatGPTWordCount: getWordCount(inputs[s1 + i*2].element.value),
    chatGPTAnswer: inputs[s1 + i*2].element.value,
    paraphraseWordCount: getWordCount(inputs[s1 + i*2 +1].element.value),
    paraphrase: inputs[s1 + i*2 +1].element.value
  }));

  const resp3 = session3Questions[language].map((q,i) => ({
    session: 3,
    question: q,
    chatGPTWordCount: getWordCount(inputs[s1 + s2 + i*2].element.value),
    chatGPTAnswer: inputs[s1 + s2 + i*2].element.value,
    retypeWordCount: getWordCount(inputs[s1 + s2 + i*2 +1].element.value),
    retype: inputs[s1 + s2 + i*2 +1].element.value
  }));

  const allResponses = [...resp1, ...resp2, ...resp3];
  const respBlob = new Blob([JSON.stringify({ responses: allResponses }, null, 2)], { type: 'application/json' });
  createDownloadLink(respBlob, 'responses_data.json');

  // UserInformation.json, including Vietnamese proficiency (#4)
  const ui = {
    gender: document.querySelector('input[name="gender"]:checked').value,
    age: document.getElementById('age').value,
    handedness: document.querySelector('input[name="handedness"]:checked').value,
    vietnameseProficiency: document.getElementById('vietnamese-proficiency').value,
    educationLevel: document.getElementById('education-level').value
  };
  const uiBlob = new Blob([JSON.stringify(ui, null, 2)], { type: 'application/json' });
  createDownloadLink(uiBlob, 'UserInformation.json');

  showThankYouMessage(respBlob, uiBlob);
}

function getWordCount(text) {
  return text.trim().split(/\s+/).filter(w => w).length;
}

function showThankYouMessage(respBlob, uiBlob) {
  const c = document.getElementById('container');
  c.innerHTML = language==='en'
    ? '<h2>Thank you for your participation!</h2><p>Please email the following JSON files to axn001@bucknell.edu</p>'
    : '<h2>Cám ơn bạn đã tham gia!</h2><p>Hãy gửi file JSON cho axn001@bucknell.edu</p>';

  const list = document.createElement('ul');
  list.innerHTML = `
    <li>responses_data.json</li>
    <li>UserInformation.json</li>
  `;
  c.appendChild(list);

  const buttons = document.createElement('div');
  buttons.className = 'button-container';
  buttons.appendChild(createManualDownloadButton(respBlob, 'responses_data.json', language==='en' ? 'Download Responses Data' : 'Tải dữ liệu trả lời'));
  buttons.appendChild(createManualDownloadButton(uiBlob,   'UserInformation.json', language==='en' ? 'Download User Information' : 'Tải dữ liệu người tham gia'));
  c.appendChild(buttons);

  const fb = document.createElement('div');
  fb.className = 'feedback';
  fb.textContent = language==='en'
    ? 'Your responses have been recorded successfully.'
    : 'Câu trả lời của bạn đã được lưu lại thành công.';
  c.appendChild(fb);
}

function createManualDownloadButton(blob, filename, text) {
  const url = URL.createObjectURL(blob);
  const btn = document.createElement('button');
  btn.className = 'btn center-text';
  btn.textContent = text;
  btn.addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  });
  return btn;
}

// ----------------------------------------------------------------------------
// 7) Alerts, popup errors & language switching
// ----------------------------------------------------------------------------
function showAlert(msg) {
  let a = document.querySelector('.alert');
  if (!a) {
    a = document.createElement('div');
    a.className = 'alert';
    document.querySelector('.container').appendChild(a);
  }
  a.textContent = msg;
  a.style.display = 'block';
}
function hideAlert() {
  const a = document.querySelector('.alert');
  if (a) a.style.display = 'none';
}
function displayPopupError(msg) {
  let p = document.getElementById('error-popup');
  if (!p) {
    p = document.createElement('div');
    p.id = 'error-popup';
    p.className = 'error-popup';
    document.body.appendChild(p);
  }
  p.textContent = msg;
  p.style.display = 'block';
}
function hidePopupError() {
  const p = document.getElementById('error-popup');
  if (p) p.style.display = 'none';
}
function hideLanguageSelection() {
  document.getElementById('language-selection').style.display = 'none';
}

// ----------------------------------------------------------------------------
// 8) Validate user info, start first session
// ----------------------------------------------------------------------------
document.getElementById('participateButton').addEventListener('click', () => {
  hideLanguageSelection();
  document.getElementById('introduction').style.display = 'none';
  document.getElementById('user-info-form').style.display = 'block';
});

document.getElementById('userInfoForm').addEventListener('submit', e => {
  e.preventDefault();
  let err = '';
  if (!document.querySelector('input[name="gender"]:checked')) {
    err += language==='en' ? 'Gender is required. ' : 'Giới tính là bắt buộc. ';
  }
  const age = document.getElementById('age').value.trim();
  if (!age) {
    err += language==='en' ? 'Age is required. ' : 'Tuổi là bắt buộc. ';
  } else if (!/^\d+$/.test(age) || +age<5 || +age>90) {
    err += language==='en' ? 'Please enter a valid age between 5 and 90. ' : 'Hãy điền tuổi của mình từ 5-90 tuổi. ';
  }
  if (!document.querySelector('input[name="handedness"]:checked')) {
    err += language==='en' ? 'Handedness is required. ' : 'Tay thuận là bắt buộc. ';
  }
  if (err) {
    displayPopupError(err);
  } else {
    hidePopupError();
    startSession();
  }
});


// ----------------------------------------------------------------------------
// 10) Event delegation for keystroke & word-count
// ----------------------------------------------------------------------------
function initializeEventListeners() {
  document.addEventListener('keydown', handleEvent);
  document.addEventListener('keyup',   handleEvent);
  document.addEventListener('input',   handleEvent);
}
function handleEvent(event) {
  if (event.target.classList.contains('input')) {
    if (event.type==='keydown' || event.type==='keyup') {
      logKeystroke(event);
    } else if (event.type==='input') {
      const wc = event.target.nextElementSibling;
      if (wc && wc.classList.contains('word-count')) {
        updateWordCount(event.target, wc);
      }
    }
  }
}
document.addEventListener('DOMContentLoaded', () => {
  initializeEventListeners();
});
function setLanguage(lang) {
  language = lang;
  translateContent(lang);
  hideLanguageSelection();
}



// Function to translate content based on selected language
function translateContent(lang) {
  document.getElementById('title').textContent = lang === 'en' ? 'Keystroke Dynamics Research' : 'Nghiên cứu về thao tác đánh máy';

  // Translate introduction paragraphs
  document.getElementById('introduction-text').textContent = lang === 'en' ?
    'Traditional plagiarism detection tools, which primarily rely on direct comparisons between a user’s input and existing sources, often struggle to identify more sophisticated forms of cheating, such as extensive paraphrasing or the use of external assistance, including generative AI or other individuals.' :
    'Các công cụ phát hiện đạo văn truyền thống, chủ yếu dựa vào việc so sánh trực tiếp giữa nội dung đầu vào của người dùng và các nguồn hiện có, thường gặp khó khăn trong việc xác định các hình thức gian lận tinh vi hơn, chẳng hạn như diễn đạt lại quá nhiều hoặc sử dụng sự hỗ trợ bên ngoài, bao gồm AI tạo ra nội dung hoặc các cá nhân khác.';

  document.getElementById('objective-text').textContent = lang === 'en' ?
    'Thus, this study aims to address academic dishonesty in writing by analyzing typing patterns and examining the differences in typing dynamics when individuals write directly compared to when they refer to or copy responses from ChatGPT. These differences are characterized by variations in thinking time, typing speed, and the frequency of editing actions during the writing process.' :
    'Do đó, nghiên cứu này nhằm mục đích giải quyết sự gian lận trong học thuật khi viết bằng cách phân tích các mẫu dữ liệu thao tác đánh máy và xem xét sự khác biệt trong thao tác đánh máy khi cá nhân viết trực tiếp so với khi họ tham khảo hoặc sao chép phản hồi từ ChatGPT. Những khác biệt này được đặc trưng bởi sự thay đổi về thời gian suy nghĩ, tốc độ đánh máy và tần suất các hành động chỉnh sửa trong quá trình viết.';

  // Translate data collection process
  document.getElementById('data-collection-process').textContent = lang === 'en' ?
    'Data Collection Process:' :
    'Quá trình thu thập dữ liệu:';
  document.getElementById('data-collection-description').textContent = lang === 'en' ?
    'There are three different sessions for collecting data. In each session, participants will respond to six questions using 90-150 words each, which are designed to invoke various cognitive load levels.' :
    'Có ba phần khác nhau để thu thập dữ liệu. Trong mỗi phần, người tham gia sẽ trả lời sáu câu hỏi bằng Dùng không quá 150 từ mỗi câu, được thiết kế để khơi gợi các mức độ nhận thức khác nhau.';
  document.getElementById('session1-description').textContent = lang === 'en' ?
    'In this session, participants need to generate responses to each question independently, without any external assistance.' :
    'Trong phần này, người tham gia cần tự trả lời câu hỏi và không có sự trợ giúp từ ngoài.';
  document.getElementById('session2-description').textContent = lang === 'en' ?
    'Paraphrasing ChatGPT Session: In this session, participants will feed each question to ChatGPT, then paraphrase the generated response. Paraphrasing is the act of restating a piece of text in your own words while retaining the original meaning.' :
    'Phần diễn giải ChatGPT: trong phần này, người tham gia gửi câu hỏi cho ChatGPT và diễn giải lại dựa câu trả lời nhận lại. Diễn giải lại (paraphrasing) là quá trình viết lại câu bằng từ ngữ khác nhưng vẫn giữ nguyên ý nghĩa cũ của câu';
  document.getElementById('session3-description').textContent = lang === 'en' ?
    'Retyping ChatGPT Session: In this session, participants will feed each prompt to ChatGPT, then retype the generated response, focusing on accurately transcribing the provided answers.' :
    'Phần đánh máy lại ChatGPT: trong phần này, người tham gia gửi câu hỏi cho ChatGPT và đánh máy lại nguyên câu trả lời, tập trung vào việc chép đúng câu trả lời. ';

  // Translate evaluation criteria
  document.getElementById('evaluation-criteria').textContent = lang === 'en' ?
    'Evaluation Criteria:' :
    'Chỉ tiêu đánh giá:';
  document.getElementById('evaluation-description').textContent = lang === 'en' ?
    'Upon submission, participant responses will be evaluated based on several criteria:' :
    'Sau khi đánh giá, người tham gia sẽ được đánh giá theo một vài chỉ tiêu:';
  document.getElementById('grammatical-accuracy').textContent = lang === 'en' ?
    'Grammatical Accuracy' :
    'Đúng ngữ pháp';
  document.getElementById('relevance').textContent = lang === 'en' ?
    'Relevance' :
    'Độ liên quan';
  document.getElementById('length').textContent = lang === 'en' ?
    'Length' :
    'Độ dài';

  // Translate violation note
  const violationNote = document.getElementById('violation-note');
  if (violationNote) {
    violationNote.textContent = lang === 'en' ?
      'Significant violations of the above could result in a reduced amount of payment for participating in this data collection.' :
      'Vi phạm tới những chỉ tiêu ở trên có thể dẫn tới việc cắt giảm tiền thưởng cho việc tham gia vào nghiên cứu này,.';
  }

  document.getElementById('participateButton').textContent = lang === 'en' ? 'Proceed to User Information' : 'Di chuyển tới phần thông tin người tham gia';
  document.getElementById('user-info-title').textContent = lang === 'en' ? 'Please provide your information to proceed:' : 'Nhập thông tin của bạn để tiếp tục:';
  document.getElementById('gender-label').innerHTML = lang === 'en' ? 'Gender:<span class="required">*</span>' : 'Giới tính:<span class="required">*</span>';
  document.getElementById('male-label').textContent = lang === 'en' ? 'Male' : 'Nam';
  document.getElementById('female-label').textContent = lang === 'en' ? 'Female' : 'Nữ';
  document.getElementById('other-label').textContent = lang === 'en' ? 'Other' : '';
  document.getElementById('age-label').innerHTML = lang === 'en' ? 'Age:<span class="required">*</span>' : 'Tuổi:<span class="required">*</span>';
  document.getElementById('handedness-label').innerHTML = lang === 'en' ? 'Handedness:<span class="required">*</span>' : 'Tay thuận:<span class="required">*</span>';
  document.getElementById('right-handed-label').textContent = lang === 'en' ? 'Right-handed' : 'Thuận tay phải';
  document.getElementById('left-handed-label').textContent = lang === 'en' ? 'Left-handed' : 'Thuận tay trái';
  document.getElementById('vietnamese-proficiency-label').textContent = lang === 'en' ? 'Proficiency in Vietnamese:' : 'Khả năng tiếng Việt:';
  document.getElementById('choose-label').textContent = lang === 'en' ? 'Choose' : 'Chọn';
  document.getElementById('expert-label').textContent = lang === 'en' ? 'Expert/Native' : 'Cao cấp/ bản xứ';
  document.getElementById('standard-label').textContent = lang === 'en' ? 'Standard' : 'Cơ bản';
  document.getElementById('average-label').textContent = lang === 'en' ? 'Average' : 'Trung bình';
  document.getElementById('education-level-label').textContent = lang === 'en' ? 'Education Level:' : 'Trình độ học vấn:';
  document.getElementById('choose-education-label').textContent = lang === 'en' ? 'Choose' : 'Chọn';
  document.getElementById('highschool-label').textContent = lang === 'en' ? 'High School' : 'Cấp Trung học Phổ Thông';
  document.getElementById('undergraduate-label').textContent = lang === 'en' ? 'Undergraduate' : 'Cấp Đại Học';
  document.getElementById('graduate-label').textContent = lang === 'en' ? 'Graduate' : 'Đã tốt nghiệp';
  document.getElementById('proceed-button').textContent = lang === 'en' ? 'Proceed to Questions' : 'Tiếp tục tới phần câu hỏi';
  document.getElementById('error-message').textContent = ''; // Clear the error message when changing the language
}


// Function to initialize event listeners for input fields using event delegation
function initializeEventListeners() {
  // Attach event listeners to the document
  document.addEventListener('keydown', handleEvent);
  document.addEventListener('keyup', handleEvent);
  document.addEventListener('input', handleEvent);
}

// Event handler function for delegated events
function handleEvent(event) {
  // Check if the event target is an input field with the class 'input'
  if (event.target.classList.contains('input')) {
    // Call the appropriate function based on the event type
    if (event.type === 'keydown' || event.type === 'keyup') {
      logKeystroke(event);
    } else if (event.type === 'input') {
      // Update word count if the event is 'input'
      let wordCountDiv = event.target.nextElementSibling;
      if (wordCountDiv && wordCountDiv.classList.contains('word-count')) {
        updateWordCount(event.target, wordCountDiv);
      }
    }
  }
}