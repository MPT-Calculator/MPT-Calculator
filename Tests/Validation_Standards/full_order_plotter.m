orig = "MPT-Calculator-paul-march/Results/";
new = "MPT-Calculator-March_Release/Results/";

%% Bomblet:
filename = 'OCC_bomblet/al_0.001_mu_16_sig_1e6/1e1-1e8_40_el_28571_ord_3_POD_13_1e-6/Data/Eigenvalues.csv';

% orig
orig_file = readmatrix(orig + filename, 'Delimiter',', ', TrimNonNumeric=true);
new_file = readmatrix(new + filename, 'Delimiter',', ', TrimNonNumeric=true);

figure;
semilogx(logspace(1,8,40), real(orig_file(:,1)), 'b')
hold on
semilogx(logspace(1,8,40), real(new_file(:,1)), 'r')
xlabel('omega')
ylabel('$\lambda_1 (re(\mathcal{M}))$', Interpreter='latex')
legend('old version', 'new version')
title('Bomblet')

figure;
semilogx(logspace(1,8,40), imag(orig_file(:,1)), 'b')
hold on
semilogx(logspace(1,8,40), imag(new_file(:,1)), 'r')
xlabel('omega')
ylabel('$\lambda_1 (im(\mathcal{M}))$', Interpreter='latex')
legend('old version', 'new version')
title('Bomblet')

rel_error = abs(new_file - orig_file) ./ abs(orig_file);
figure;
loglog(logspace(1,8,40), rel_error)
title('Bomblet')
xlabel('omega')
ylabel('rel error, eigenvalues')


%% DualBar:
filename = 'OCC_dualbar/al_0.001_mu_1,1_sig_1e6,1e8/1e1-1e8_40_el_78714_ord_3_POD_13_1e-6/Data/Eigenvalues.csv';
PODSnapshots = 'OCC_dualbar/al_0.001_mu_1,1_sig_1e6,1e8/1e1-1e8_40_el_78714_ord_3_POD_13_1e-6/Data/PODEigenvalues.csv';

% orig
orig_file = readmatrix(orig + filename, 'Delimiter',', ', TrimNonNumeric=true);
new_file = readmatrix(new + filename, 'Delimiter',', ', TrimNonNumeric=true);
orig_file_POD = readmatrix(orig + PODSnapshots, 'Delimiter',', ', TrimNonNumeric=true);
new_file_POD = readmatrix(new + PODSnapshots, 'Delimiter',', ', TrimNonNumeric=true);

figure;
semilogx(logspace(1,8,40), real(orig_file(:,1)), 'b')
hold on
semilogx(logspace(1,8,40), real(new_file(:,1)), 'r')
scatter(logspace(1,8,13), real(orig_file_POD(:,1)), 'b*')
scatter(logspace(1,8,13), real(new_file_POD(:,1)), 'r*')
xlabel('omega')
ylabel('$\lambda_1 (re(\mathcal{M}))$', Interpreter='latex')
legend('old version', 'new version')
title('dual bar')
axis tight

figure;
semilogx(logspace(1,8,40), imag(orig_file(:,1)), 'b')
hold on
semilogx(logspace(1,8,40), imag(new_file(:,1)), 'r')
scatter(logspace(1,8,13), imag(orig_file_POD(:,1)), 'b*')
scatter(logspace(1,8,13), imag(new_file_POD(:,1)), 'r*')
xlabel('omega')
ylabel('$\lambda_1 (im(\mathcal{M}))$', Interpreter='latex')
legend('old version', 'new version')
title('dual bar')
axis tight

rel_error = abs(new_file - orig_file) ./ abs(orig_file);
figure;
loglog(logspace(1,8,40), rel_error)
title('dual bar ')
xlabel('omega')
ylabel('rel error, eigenvalues')
legend('$\lambda_1$', '$\lambda_2$', '$\lambda_3$', Interpreter='latex')
axis tight

%% Key 4:
filename = 'OCC_key_4/al_0.001_mu_141.3135696662735_sig_1.5e7/1e1-1e8_40_el_39128_ord_3_POD_13_1e-6/Data/Eigenvalues.csv';
PODSnapshots = 'OCC_key_4/al_0.001_mu_141.3135696662735_sig_1.5e7/1e1-1e8_40_el_39128_ord_3_POD_13_1e-6/Data/PODEigenvalues.csv';


% orig
orig_file = readmatrix(orig + filename, 'Delimiter',', ', TrimNonNumeric=true);
new_file = readmatrix(new + filename, 'Delimiter',', ', TrimNonNumeric=true);
orig_file_POD = readmatrix(orig + PODSnapshots, 'Delimiter',', ', TrimNonNumeric=true);
new_file_POD = readmatrix(new + PODSnapshots, 'Delimiter',', ', TrimNonNumeric=true);

figure;
semilogx(logspace(1,8,40), real(orig_file(:,1)), 'b')
hold on
semilogx(logspace(1,8,40), real(new_file(:,1)), 'r')
scatter(logspace(1,8,13), real(orig_file_POD(:,1)), 'b*')
scatter(logspace(1,8,13), real(new_file_POD(:,1)), 'r*')
xlabel('omega')
ylabel('$\lambda_1 (re(\mathcal{M}))$', Interpreter='latex')
legend('old version', 'new version')
title('key 4')
axis tight

figure;
semilogx(logspace(1,8,40), imag(orig_file(:,1)), 'b')
hold on
semilogx(logspace(1,8,40), imag(new_file(:,1)), 'r')
scatter(logspace(1,8,13), imag(orig_file_POD(:,1)), 'b*')
scatter(logspace(1,8,13), imag(new_file_POD(:,1)), 'r*')
xlabel('omega')
ylabel('$\lambda_1 (im(\mathcal{M}))$', Interpreter='latex')
legend('old version', 'new version')
title('key 4')
axis tight

rel_error = abs(new_file - orig_file) ./ abs(orig_file);
figure;
loglog(logspace(1,8,40), rel_error)
title('key 4')
xlabel('omega')
ylabel('rel error, eigenvalues')
legend('$\lambda_1$', '$\lambda_2$', '$\lambda_3$', Interpreter='latex')
axis tight

%% Thin Disk Magnetic:

filename = 'OCC_thin_disc_magnetic_32/al_0.001_mu_32_sig_1e6/1e1-1e8_40_el_27743_ord_3_POD_13_1e-6/Data/Eigenvalues.csv';
PODSnapshots = 'OCC_thin_disc_magnetic_32/al_0.001_mu_32_sig_1e6/1e1-1e8_40_el_27743_ord_3_POD_13_1e-6/Data/PODEigenvalues.csv';


% orig
orig_file = readmatrix(orig + filename, 'Delimiter',', ', TrimNonNumeric=true);
new_file = readmatrix(new + filename, 'Delimiter',', ', TrimNonNumeric=true);
orig_file_POD = readmatrix(orig + PODSnapshots, 'Delimiter',', ', TrimNonNumeric=true);
new_file_POD = readmatrix(new + PODSnapshots, 'Delimiter',', ', TrimNonNumeric=true);

figure;
semilogx(logspace(1,8,40), real(orig_file(:,1)), 'b')
hold on
semilogx(logspace(1,8,40), real(new_file(:,1)), 'r')
scatter(logspace(1,8,13), real(orig_file_POD(:,1)), 'b*')
scatter(logspace(1,8,13), real(new_file_POD(:,1)), 'r*')
xlabel('omega')
ylabel('$\lambda_1 (re(\mathcal{M}))$', Interpreter='latex')
legend('old version', 'new version')
title('Thin Magnetic Disk')
axis tight

figure;
semilogx(logspace(1,8,40), imag(orig_file(:,1)), 'b')
hold on
semilogx(logspace(1,8,40), imag(new_file(:,1)), 'r')
scatter(logspace(1,8,13), imag(orig_file_POD(:,1)), 'b*')
scatter(logspace(1,8,13), imag(new_file_POD(:,1)), 'r*')
xlabel('omega')
ylabel('$\lambda_1 (im(\mathcal{M}))$', Interpreter='latex')
legend('old version', 'new version')
title('Thin Magnetic Disk')
axis tight

rel_error = abs(new_file - orig_file) ./ abs(orig_file);
figure;
loglog(logspace(1,8,40), rel_error)
title('Thin Magnetic Disk')
xlabel('omega')
ylabel('rel error, eigenvalues')
legend('$\lambda_1$', '$\lambda_2$', '$\lambda_3$', Interpreter='latex')
axis tight


%% Sphere w. Prisms mur=32:
filename = 'OCC_sphere_prism_32/al_0.01_mu_1_sig_1e6/1e1-1e8_40_el_22426_ord_3_POD_13_1e-6/Data/Eigenvalues.csv';
PODSnapshots = 'OCC_sphere_prism_32/al_0.01_mu_1_sig_1e6/1e1-1e8_40_el_22426_ord_3_POD_13_1e-6/Data/PODEigenvalues.csv';

% orig
orig_file = readmatrix(orig + filename, 'Delimiter',', ', TrimNonNumeric=true);
new_file = readmatrix(new + filename, 'Delimiter',', ', TrimNonNumeric=true);
orig_file_POD = readmatrix(orig + PODSnapshots, 'Delimiter',', ', TrimNonNumeric=true);
new_file_POD = readmatrix(new + PODSnapshots, 'Delimiter',', ', TrimNonNumeric=true);


figure;
semilogx(logspace(1,8,40), real(orig_file(:,1)), 'b')
hold on
semilogx(logspace(1,8,40), real(new_file(:,1)), 'r')
scatter(logspace(1,8,13), real(orig_file_POD(:,1)), 'b*')
scatter(logspace(1,8,13), real(new_file_POD(:,1)), 'r*')
xlabel('omega')
ylabel('$\lambda_1 (re(\mathcal{M}))$', Interpreter='latex')
legend('old version', 'new version')
title('Magnetic Sphere')
axis tight

figure;
semilogx(logspace(1,8,40), imag(orig_file(:,1)), 'b')
hold on
semilogx(logspace(1,8,40), imag(new_file(:,1)), 'r')
scatter(logspace(1,8,13), imag(orig_file_POD(:,1)), 'b*')
scatter(logspace(1,8,13), imag(new_file_POD(:,1)), 'r*')
xlabel('omega')
ylabel('$\lambda_1 (im(\mathcal{M}))$', Interpreter='latex')
legend('old version', 'new version')
title('Magnetic Sphere')
axis tight

rel_error = abs(new_file - orig_file) ./ abs(orig_file);
figure;
loglog(logspace(1,8,40), rel_error)
title('Magnetic Sphere')
xlabel('omega')
ylabel('rel error, eigenvalues')
legend('$\lambda_1$', '$\lambda_2$', '$\lambda_3$', Interpreter='latex')
axis tight
