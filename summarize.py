import json
d=json.load(open('results.json'))
print('rows',len(d))
for r in d:
    if r.get('status')!='ok':
        print('ERR',r['case'],r['scheme'],r['backend'],r.get('error')); continue
    m=r['metrics_vs_fp64']
    eig=m.get('eig_max_abs','n/a')
    print(f"{r['case']:16s} {r['scheme']:9s} {r['backend']:3s} ozaki_ms={r['ozaki_time_ms']:7.2f} speedup={r['speedup_vs_fp64']:6.3f} maxabs={m['max_abs']:.3e} sse={m['sse']:.3e} frorel={m['fro_rel']:.3e} eig={eig}")
