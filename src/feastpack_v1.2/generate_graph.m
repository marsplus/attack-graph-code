%Taken from
%https://www.sandia.gov/~tgkolda/feastpack/doc_bter_ideal.html#13
for i = 0:99
    nnodes = 375;
    maxdeg_bound = nnodes;
    avgdeg_target = 6;
    maxccd_target = 0.98;
    gcc_target = 0.4;
    tau = 1e-3 / nnodes;
    [alpha, beta] = degdist_param_search(avgdeg_target, maxdeg_bound, 'maxdeg_prbnd', tau);
    pdf = dglnpdf(maxdeg_bound,alpha,beta);
    nd = gendegdist(nnodes, pdf)
    maxdeg = find(nd > 0, 1, 'last');
    xi = cc_param_search(nd, maxccd_target, gcc_target);
    ccd_target = [0; maxccd_target * exp(-(0:maxdeg-2)'.* xi)];
    ccd_stddev = min(0.01,ccd_target/2);
    ccd = max(ccd_target +  randn(size(ccd_target)) .* ccd_stddev,0);
    [E1,E2] = bter(nd,ccd);
    filename = strcat('../data/BTER_', num2str(i, '%02.f'), '.txt.raw')
    disp(filename)
    file = fopen(filename, 'w');
    fprintf(file, '%d %d\n', E1);
    fprintf(file, '%d %d\n', E2);
    fclose(file);
endfor
