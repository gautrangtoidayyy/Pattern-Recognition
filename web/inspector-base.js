export const descriptions={
 raw:['Một chữ số, 784 pixel','Ảnh gốc MNIST 28×28. Mỗi pixel có cường độ từ 0 đến 1. Nhãn thật chỉ dùng để đối chiếu, không truyền vào mô hình.','xᵢ = pixelᵢ / 255'],
 normalize:['Đưa dữ liệu về đúng miền huấn luyện','Gaussian Bayes, Logistic, MLP và AdaBoost dùng Z-score. CNN và Multinomial Bayes dùng pixel [0,1], đúng dữ liệu huấn luyện mới.','xᵢ = (pixelᵢ / 255 − μ_train) / σ_train'],
 bayes:['Mỗi pixel đóng góp thế nào cho một lớp?','Chọn lớp 0–9 để xem log-likelihood của 784 pixel. Gaussian dùng trung bình và phương sai đã học. Multinomial dùng tần suất pixel và Laplace α.','Gaussian: −½ log(2πσ²) − (x − μ)² / (2σ²)\nMultinomial: xᵢ log((countᵢ + α) / Σ(count + α))'],
 prior:['Xác suất tiên nghiệm từ tập train','Mỗi lớp có một tần suất trong 50.000 ảnh train. Log-prior được cộng vào tổng log-likelihood.','log P(C = c)'],
 posterior:['Cộng 784 bằng chứng và log-prior','Mô hình dự đoán lớp có điểm log-posterior lớn nhất. Bước sau chuẩn hóa các điểm bằng softmax để hiển thị xác suất.','score_c = log P(c) + Σᵢ log P(xᵢ | c)'],
 lrterms:['Trọng số cho thấy bằng chứng của từng pixel','Đổi lớp để xem bản đồ đóng góp. Giá trị dương làm tăng điểm lớp đang xem; giá trị âm làm giảm điểm.','contributionᵢ,c = xᵢ × Wᵢ,c'],
 lrlogits:['Một tổng có trọng số cho mỗi chữ số','784 đóng góp được cộng với bias. Checkpoint gốc lưu bias ở hàng cuối của W (785×10).','z_c = Σᵢ xᵢ Wᵢ,c + W₇₈₄,c'],
 linear:['Từng neuron là một tổng có trọng số','Chọn neuron để xem bias, tổng và các đóng góp lớn nhất theo trị tuyệt đối. Bảng chỉ rút gọn phần hiển thị; kết quả dùng đủ mọi đầu vào.','z_j = Σᵢ aᵢ W_j,i + b_j'],
 relu:['Giữ tín hiệu dương, chặn tín hiệu âm','ReLU được áp dụng độc lập cho từng phần tử. Giá trị âm trở thành 0, còn giá trị dương giữ nguyên.','aᵢ = max(0, zᵢ)'],
 conv:['Bộ lọc học được quét trên ảnh thật','Kernel 3×3, stride 1, padding 1. Chọn kênh và pixel để kiểm tra các tích, tổng và bias.','z[c,y,x] = b[c] + Σₖ,dy,dx input[k,y+dy−1,x+dx−1] × W[c,k,dy,dx]'],
 pool:['Giữ giá trị lớn nhất trong vùng 2×2','Max pooling 2×2, stride 2.','p[c,y,x] = max(a[c,2y:2y+2,2x:2x+2])'],
 flatten:['Trải phẳng mà không thay đổi giá trị','784 giá trị được đọc theo thứ tự channel → row → column (CHW).','index = channel × 49 + row × 7 + column'],
 trees:['200 cây cùng đưa ra dự đoán','Mỗi cây quyết định có độ sâu tối đa 3. Chọn một cây để xem đường đi: pixel nào được so với ngưỡng nào và nhánh nào được chọn.','hₜ(x): đi từ gốc đến lá theo x[feature] ≤ threshold'],
 votes:['Theo dõi phiếu bầu qua từng cây','Mỗi hàng là tổng phiếu sau một cây, mỗi cột là một lớp. SAMME cộng αₜ cho lớp được chọn và −αₜ/9 cho các lớp còn lại.','Sₜ,c = Sₜ₋₁,c + αₜ × (1 nếu hₜ(x)=c, ngược lại −1/9)'],
 samme:['Chuẩn hóa tổng phiếu có trọng số','Điểm của mỗi lớp chia cho tổng trọng số 200 cây. Đây là decision_function của AdaBoost SAMME trong scikit-learn.','f_c = S₂₀₀,c / Σₜ αₜ'],
 softmax:['Chuyển điểm số thành xác suất','Softmax ổn định số học bằng cách trừ điểm lớn nhất. Với AdaBoost đa lớp, điểm SAMME được chia thêm cho K−1 = 9 trước softmax.','P(c|x) = exp(z_c − max(z)) / Σₖ exp(z_k − max(z))']
};

export function createInspector({state,getNet,$,fmt,plane,current}){
function table(rows,headers=['i','đầu vào','trọng số','tích']){return'<table><thead><tr>'+headers.map(v=>'<th>'+v+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')+'</tbody></table>';}
function inspect(){const net=getNet();const s=current(),p=plane(),i=Math.min(state.cell,p.values.length-1),global=p.offset+i,v=s.data[global],row=Math.floor(i/p.cols),col=i%p.cols;$('inspectLabel').textContent=s.op==='trees'?'Cây #'+i:'Phần tử #'+global;$('inspectValue').textContent=fmt(v,7);$('inspectPosition').textContent=s.shape.length===3?`kênh ${state.channel} · hàng ${row} · cột ${col}`:`hàng ${row} · cột ${col}`;
 let html='',terms=[];
 if(s.op==='raw')html=`<p>Cường độ pixel: <code>${Math.round(v*255)} / 255 = ${fmt(v,7)}</code></p>`;
 if(s.op==='normalize'&&state.model==='bnb')html=`<p>Pixel ${fmt(state.result.raw[i],7)} ${v?' > ':' ≤ '} 0,5 → <code>${v}</code></p>`;
 else if(s.op==='normalize')html=`<p>${['mnb','cnn'].includes(state.model)?'Min–max':'Z-score'}<code>${['mnb','cnn'].includes(state.model)?fmt(v,7):'('+fmt(state.result.raw[i],7)+' − '+fmt(net.meta.mean,7)+') / '+fmt(net.meta.std,7)+' = '+fmt(v,7)}</code></p><p>Đầu vào được lưu float32, cùng kiểu dữ liệu của tập test gốc.</p>`;
 if(s.op==='conv'){
  const [channels,side]=s.inputShape,k=s.kernelSize||3,pad=s.padding||0;let sum=0;for(let ic=0;ic<channels;ic++)for(let ky=0;ky<k;ky++)for(let kx=0;kx<k;kx++){let y=row+ky-pad,x=col+kx-pad,a=y>=0&&y<side&&x>=0&&x<side?s.input[ic*side*side+y*side+x]:0,w=s.weights[(state.channel*channels+ic)*k*k+ky*k+kx];sum+=a*w;terms.push([`${ic}:${ky},${kx}`,fmt(a,4),fmt(w,4),fmt(a*w,5)]);}html=`<p>Filter #${state.channel} · ${channels} kênh · kernel ${k}×${k} · padding ${pad}</p>${table(terms)}<p>Σ ${terms.length} tích + bias <code>${fmt(sum,6)} + ${fmt(s.bias[state.channel],6)} = ${fmt(v,6)}</code></p>`;
 }
 if(s.op==='linear'){
  for(let j=0;j<s.input.length;j++){const w=s.weights[i*s.input.length+j];terms.push([j,s.input[j],w,s.input[j]*w]);}terms.sort((a,b)=>Math.abs(b[3])-Math.abs(a[3]));html=`<p>Neuron #${i} · ${s.input.length} đầu vào</p><code>Σ aᵢwᵢ + b = ${fmt(v-s.bias[i],6)} + ${fmt(s.bias[i],6)} = ${fmt(v,6)}</code><p>8 đóng góp lớn nhất theo |aᵢwᵢ|:</p>${table(terms.slice(0,8).map(r=>[r[0],...r.slice(1).map(x=>fmt(x,5))]))}`;
 }
 if(s.op==='relu')html=`<p>Đầu vào z = ${fmt(s.input[global],7)}</p><code>max(0, ${fmt(s.input[global],7)}) = ${fmt(v,7)}</code>`;
 if(s.op==='pool'){const side=s.inputShape[1],base=state.channel*side*side+row*2*side+col*2,values=[s.input[base],s.input[base+1],s.input[base+side],s.input[base+side+1]];html=`<p>4 giá trị vùng [${row*2}:${row*2+2}, ${col*2}:${col*2+2}]:</p><code>${values.map(x=>fmt(x,6)).join(' · ')}<br>max = ${fmt(v,6)}</code>`;}
 if(s.op==='flatten')html=state.model!=='cnn'?`<p>Ảnh 28 × 28 được trải theo từng hàng, không đổi giá trị:</p><code>hàng = ${Math.floor(i/28)}<br>cột = ${i%28}<br>index = ${i}<br>giá trị ${state.model==='bnb'?'nhị phân':state.model==='mnb'?'[0,1]':'Z-score'} = ${fmt(v,7)}</code>`:`<p>Ánh xạ CHW:</p><code>kênh = ${Math.floor(i/49)}<br>hàng = ${Math.floor((i%49)/7)}<br>cột = ${i%7}<br>index = ${i}<br>giá trị = ${fmt(v,7)}</code>`;
 if(s.op==='bayes'){
  const c=state.channel,x=s.input[i];
  if(state.model==='bnb'){const theta=net.w['bnb.theta'][global];html=`<p>Lớp ${c}, pixel ${i}</p><code>x = ${x}<br>θ = ${fmt(theta,8)}<br>log P(x|lớp) = log(${x?'θ':'1−θ'}) = ${fmt(v,7)}</code>`;}
  else if(state.model==='gnb'){const mu=net.w['gnb.mean'][global],variance=net.w['gnb.var'][global];html=`<p>Lớp ${c}, pixel ${i}</p><code>x = ${fmt(x,7)}<br>μ = ${fmt(mu,7)}<br>σ² = ${fmt(variance,7)}<br>−½log(2πσ²) = ${fmt(-.5*Math.log(2*Math.PI*variance),7)}<br>−(x−μ)²/(2σ²) = ${fmt(-((x-mu)**2)/(2*variance),7)}<br>tổng = ${fmt(v,7)}</code>`;}
  else {let den=0;for(let j=0;j<784;j++)den+=net.w['mnb.feature_count'][c*784+j]+net.w['mnb.alpha'][0];const count=net.w['mnb.feature_count'][global],alpha=net.w['mnb.alpha'][0],theta=(count+alpha)/den;html=`<code>x = ${fmt(x,7)}<br>count = ${fmt(count,7)}<br>α = ${alpha}<br>Σ(count+α) = ${fmt(den,7)}<br>θ = ${fmt(theta,10)}<br>x × log θ = ${fmt(v,7)}</code>`;}
 }
 if(s.op==='prior')html=`<code>P(${i}) = ${fmt(net.w[state.model+'.prior'][i],7)}<br>log P(${i}) = ${fmt(v,7)}</code>`;
 if(s.op==='posterior'){const prior=Math.log(net.w[state.model+'.prior'][i]);html=`<p>Lớp ${i}</p><code>Σ log-likelihood = ${fmt(v-prior,6)}<br>log-prior = ${fmt(prior,6)}<br>điểm = ${fmt(v,6)}</code>`;}
 if(s.op==='lrterms'){const w=net.w['lr.W'][i*10+state.channel];html=`<code>x[${i}] = ${fmt(s.input[i],7)}<br>W[${i},${state.channel}] = ${fmt(w,7)}<br>x × W = ${fmt(v,7)}</code>`;}
 if(s.op==='lrlogits'){const bias=net.w['lr.W'][784*10+i];html=`<code>Σ 784 tích = ${fmt(v-bias,7)}<br>bias = ${fmt(bias,7)}<br>logit = ${fmt(v,7)}</code>`;}
 if(s.op==='softmax'){const z=s.ada?s.input.map(x=>x/9):s.input,m=Math.max(...z),den=z.reduce((a,b)=>a+Math.exp(b-m),0);html=`<p>Lớp ${i}${s.ada?' · điểm SAMME / 9':''}</p><code>z = ${fmt(z[i],7)}<br>max(z) = ${fmt(m,7)}<br>exp(z − max) = ${fmt(Math.exp(z[i]-m),9)}<br>Σ exp = ${fmt(den,9)}<br>P = ${fmt(v,9)}</code>`;}
 if(s.op==='trees'){html=`<p>Cây #${i} · α = ${fmt(net.weights[i],6)} · lỗi train có trọng số ${fmt(net.errors[i],6)}</p>`+s.paths[i].map((n,j)=>n.pred!==undefined?`<code>→ Lá ${n.node}: dự đoán ${n.pred}</code>`:`<p>${j+1}. Node ${n.node}, pixel ${n.feature} (hàng ${Math.floor(n.feature/28)}, cột ${n.feature%28})</p><code>${fmt(n.value,6)} ≤ ${fmt(n.threshold,6)} → ${n.left?'ĐÚNG · trái':'SAI · phải'}</code>`).join('');}
 if(s.op==='votes'){const tree=Math.floor(i/10),cls=i%10,prev=tree?s.data[(tree-1)*10+cls]:0,pred=state.result.stages.find(x=>x.op==='trees').data[tree],w=net.weights[tree];html=`<p>Sau cây #${tree}, lớp ${cls}</p><code>cây dự đoán ${pred}<br>α = ${fmt(w,7)}<br>trước = ${fmt(prev,7)}<br>cộng ${fmt(pred===cls?w:-w/9,7)}<br>tổng = ${fmt(v,7)}</code>`;}
 if(s.op==='samme'){const total=net.weights.reduce((a,b)=>a+b,0);html=`<code>Σ α = ${fmt(total,7)}<br>phiếu lớp ${i} = ${fmt(v*total,7)}<br>điểm = ${fmt(v,7)}</code>`;}
 $('mathDetail').innerHTML=html;
}

return inspect;
}
