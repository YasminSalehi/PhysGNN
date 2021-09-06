
%% Note: 
%  This code has been adapted from the templates provided in the Gibbon 
%  documentation listed below:  
%  1) DEMO_febio_0060_vertebrae_disc_01.m
%  2) DEMO_febio_0063_custom_hip_implant_01.m 
%  3) DEMO_febio_0047_cylinder_embedded_probe_01.m

%%
clear; close all; clc;
%% Plot settings
fontSize=15;
faceAlpha=0.5;
lineWidth1=1.5;
lineWidth2=3;
markerSize1=15;
markerSize2=30;
edgeWidth=2;
edgeColor='k';
faceAlpha1=0.5;
% -------------------------------------------------------------------------
plot = 0; % decide to plot or not. value: 0/1
animation_displacement = 1; % decide to animate displacement or not. value: 0/1
animation_energy = 0; % decide to animate energy or not. value: 0/1
view_specs = 0; % decide to view the febio specification file or not. value: 0/1
save_var = 0; % decide to save the variables or not. value: 0/1
% -------------------------------------------------------------------------
%% Control parameters
% Path names
defaultFolder = '/PhysGNN/Data_Generator/code'; % set the path name
savePath=fullfile(defaultFolder,'data','temp_10');
pathNameSTL = fullfile(defaultFolder, 'data', 'stl');
% -------------------------------------------------------------------------
% Defining file names
febioFebFileNamePart='BTM';
febioFebFileName=fullfile(savePath,[febioFebFileNamePart,'.feb']); % FEB file name
febioLogFileName=fullfile(savePath,[febioFebFileNamePart,'.txt']); % FEBio log file name
febioLogFileName_disp=[febioFebFileNamePart,'_disp_out.txt'];      % Log file name for exporting displacement
febioLogFileName_force=[febioFebFileNamePart,'_force_out.txt'];    % Log file name for exporting force
febioLogFileName_stress=[febioFebFileNamePart,'_stress_out.txt'];  % Log file name for exporting stresses
febioLogFileName_strain=[febioFebFileNamePart,'_strain_out.txt'];  % Log file name for exporting strain
febioLogFileName_strainEnergy=[febioFebFileNamePart,'_energy_out.txt']; % Log file name for exporting strain energy density
% -------------------------------------------------------------------------
% Defining input/output folder names
input_folder_name = 'input_1';
output_folder_name = 'output_1';
% -------------------------------------------------------------------------
%% Units
% Primary units
% Length: Millimeters (mm) 
% Force: Newtons (N)
% Time: Seconds (s)
% Derived units
% Density: tonne/(m * 10^3)^3 = kg * 10^-3 / m^3 * 10^9 = [tonne/mm^3] * 10^-12
% Pressure (Young's Modulus) = N/(m * 10^3)^2 = [N/m] * 10^-6 
%% Material parameter set
% Brain
density1=1000 * (10^-12);       % Density(tonne/mm^3) 
E_youngs1=3000 * (10^-6);       % Youngs modulus (MPa)
nu1=0.49;                       % Poissons ratio
% -------------------------------------------------------------------------
% Tumour
density2=1000 * (10^-12) ;      % Density(tonne/mm^3) 
E_youngs2=7500 * (10^-6);       % Youngs modulus (MPa)
nu2=0.49;                       % Poissons ratio
% -------------------------------------------------------------------------
%% FEA control settings
f_magnitude = 1.35;             % Total force magnitude (N)
numTimeSteps=30;                % Number of time steps desired
max_refs=25;                    % Max reforms
max_ups=0;                      % Set to zero to use full-Newton iterations
opt_iter=6;                     % Optimum number of iterations
max_retries=5;                  % Maximum number of retires
dtmin=(1/numTimeSteps)/100;     % Minimum time step size
dtmax=1/numTimeSteps;           % Maximum time step size
runMode='internal';             %'external' or 'internal'
% -------------------------------------------------------------------------
%% Import brain and tumour model
% Import the brain model
[stlStruct_brain] = import_STL(fullfile(pathNameSTL,'brain.stl'));
F_brain=stlStruct_brain.solidFaces{1}; %Faces brain
V_brain=stlStruct_brain.solidVertices{1}; %Vertices brain
% [F_brain, V_brain]= mergeVertices(F_brain, V_brain); %merging nodes
% -------------------------------------------------------------------------
% Import the tumour model
[stlStruct_tumour] = import_STL(fullfile(pathNameSTL,'tumour.stl'));
F_tumour=stlStruct_tumour.solidFaces{1}; %Faces tumour
V_tumour=stlStruct_tumour.solidVertices{1}; %Vertices tumour
% [F_tumour, V_tumour]= mergeVertices(F_tumour, V_tumour); %merging nodes
%% Merge model components
[F,V,C]=joinElementSets({F_brain,F_tumour},{V_brain,V_tumour});
[F,V]=mergeVertices(F,V);
% cFigure; hold on;
% title('The Brain and Tumour Merged Surfaces');
% gpatch(F,V,C,'none',0.5);
% axisGeom; 
% camlight headlight;
% colormap(gjet(250)); icolorbar;
% gdrawnow;
%% Mesh solid using tetgen
% Create tetgen meshing input structure
[regionA]=tetVolMeanEst(F,V); % Volume for a regular tet based on edge lengths
V_inner=getInnerPoint(F,V); % Interior point for region
volumeFactor=2;
% -------------------------------------------------------------------------
inputStruct.stringOpt='-pq1.2AaY';
inputStruct.Faces=F;
inputStruct.Nodes=V;
inputStruct.holePoints=[];
inputStruct.faceBoundaryMarker=C; % Face boundary markers
inputStruct.regionPoints=V_inner; % Region points
inputStruct.regionA=regionA*volumeFactor; % Desired volume for tets
inputStruct.minRegionMarker=-1; % Minimum region marker
% -------------------------------------------------------------------------
% Mesh model using tetrahedral elements using tetGen
[meshOutput]=runTetGen(inputStruct); % Run tetGen 
% -------------------------------------------------------------------------
%% Access model element and patch data 
V=meshOutput.nodes; % The vertices or nodes
F=meshOutput.faces; % The faces
C=meshOutput.faceMaterialID; % The face id
E=meshOutput.elements; % The elements
% -------------------------------------------------------------------------
elementMaterialID=meshOutput.elementMaterialID; % Element material or region id
% -------------------------------------------------------------------------
Fb=meshOutput.facesBoundary; % The boundary faces
Cb=meshOutput.boundaryMarker; % The boundary marker
% -------------------------------------------------------------------------
%% ***SAVE #1 The FE Mesh 
if save_var == 1
    save(strcat(input_folder_name,'/mesh_output.mat'),'meshOutput');
    csvwrite(strcat(input_folder_name,'/xyz.csv'), V);
    csvwrite(strcat(input_folder_name,'/elements.csv'), E);
    csvwrite(strcat(input_folder_name,'/element_ID.csv'), elementMaterialID);
    csvwrite(strcat(input_folder_name,'/boundary_faces.csv'), Fb);
    csvwrite(strcat(input_folder_name,'/boundary_marker.csv'), Cb);
end
% -------------------------------------------------------------------------
%% Define boundary condition node sets
% Support list 
w=100; % Define an 80 x 80 plane
f=[1 2 3 4];
v=w*[-1 -1 0; -1 1 0; 1 1 0; 1 -1 0]; % v is the plane
p=[80 90 120]; % x y z translation
Q=euler2DCM([(20/180)*pi 0 0]); % Plane rotation
v=v*Q;
v=v+p;  % v is the transformed plane
% -------------------------------------------------------------------------
Vr=V*Q';
Vr=Vr+p;
logicRigidNodes=Vr(:,3)<265;
logicRigidFaces=all(logicRigidNodes(Fb),2)&(Cb(:,1)==1);
bcSupportList=unique(Fb(logicRigidFaces,:));
% -----------------------------------------------------
%% ***SAVE #2 The Support List
if save_var == 1
    save(strcat(input_folder_name,'/bcSupportList.mat'),'bcSupportList');
    csvwrite(strcat(input_folder_name,'/bcSupportList.csv'), bcSupportList);
end
% --------------------------------------------------------
%% Prescribed list
w2=80; % Define a 50x50 plane
f2=[1 2 3 4];
v2=w2*[-1 -1 0; -1 1 0; 1 1 0; 1 -1 0]; % v is the plane
p2=[80 110 130]; % x y z translation
Q2=euler2DCM([(15/180)*pi (20/180)*pi 0]); % Plane rotation
v2=v2*Q2;
v2=v2+p2; % v is the transformed plane
% -------------------------------------------------------------------------
Vr2=V*Q2';
Vr2=Vr2+p2;
% logicHeadNodes=Vr2(:,3)<=-50.4; % 100 prescribed nodes
logicHeadNodes=(Vr2(:,3)>275.5) ; % prescribed nodes
logicHeadFaces=all(logicHeadNodes(Fb),2)&(Cb(:,1)==1);
bcPrescribeList=unique(Fb(logicHeadFaces,:));
% -------------------------------------------------------------------------
%% ***SAVE #3 The Prescribed List
if save_var == 1
    save(strcat(input_folder_name,'/bcPrescribeList.mat'),'bcPrescribeList');
    csvwrite(strcat(input_folder_name,'/bcPrescribeList.csv'), bcPrescribeList);
end
% -------------------------------------------------------------------------
%% Defining the free boundary condition (for visualization only)
logicFreeNodes=Vr(:,3)>=265;
logicFreeFaces=(Cb(:,1)==2)|(all(logicFreeNodes(Fb),2));
bcFree=unique(Fb(logicFreeFaces,:));
% -------------------------------------------------------------------------
%% Visualize boundary conditions and ALL prescribed loads 
% cFigure;
% hold on;
% gpatch(Fb,V,'w','k',0.7);
% gpatch(f,v,'b','k',0.5);
% % gpatch(f2,v2,'r','k',0.5);
% hl(1)=plotV(V(bcSupportList,:),'b.','MarkerSize',30);
% hl(2)=plotV(V(bcFree,:),'g.','markerSize',40);
% hl(3)=plotV(V(bcPrescribeList,:),'r.','markerSize',25);
% legend(hl,{'BC support','BC free','BC force prescribe'}, 'FontSize',20);
% axisGeom;
% camlight headlight;
% drawnow;
% -------------------------------------------------------------------------
%% Force Distribution

[~,~,N]=patchNormal((Fb),V); %Nodal normal directions
FX=[1 0 0]; %X force vector
FY=[0 1 0]; %Y force vector
FZ=[0 0 1]; %Z force vector

force_X_norm=dot(N(bcPrescribeList,:),FX(ones(numel(bcPrescribeList),1),:),2);
force_Y_norm=dot(N(bcPrescribeList,:),FY(ones(numel(bcPrescribeList),1),:),2);
force_Z_norm=dot(N(bcPrescribeList,:),FZ(ones(numel(bcPrescribeList),1),:),2);

% disp(force_X_norm)
% disp(force_Y_norm)
% disp(force_Z_norm)

len_norms = size(bcPrescribeList,1);

% The total number of directions is equal to (iter + 1)
iter = 14; 

xyz_directions = zeros(len_norms*(iter + 1),3);

rng(0);
s = rng;
% Here we are filtering out random forces that cause negative jacobins or
% unrealistic simulations based on the selected force load nodes. The
% number "6" is an arbitrary number to generate more forces than needed,
% such that when forces are filtered out, we have the neccessary number of
% forces. **Note: the ranges selected (i.e. x<0.6, y<0.6 and 0.2 < z < 0.9
% are based on trial and error.
uv = rand((len_norms*iter*6),2);
z =  1.0 - (2.0 .* uv(:,1));
r = sqrt(1- z.*z);
phi = (2 * pi) .* uv(:,2);
random_norm = [r .* cos(phi), r .* sin(phi), z];

random_directions = [0 0 0];

for i=1:length(random_norm)
    
    rand_n_x = abs(random_norm(i,1));
    rand_n_y = abs(random_norm(i,2));
    rand_n_z = abs(random_norm(i,3));
    
    if (rand_n_x < 0.6) && (rand_n_y < 0.6) && (rand_n_z > 0.2) && (rand_n_z < 0.9)
        
        if isequal(random_directions, [0 0 0])
            random_directions = random_norm(i,:);
        else
            random_directions = cat(1,random_directions,random_norm(i,:));
        end
    end
end

random_directions = random_directions(1:(len_norms*iter),:); % ommiting the unneccessary rows.
% --------------------------------------------------------------------------------------------
% Since the above forced for node 4 (direction 2, 4, 6, 8, 9, 13, 15), node
% 5 (direction 12) and node 7 (direction 8) result in negative jacobians,
% their value are replaced by the new numbers generated below.
uv = rand((len_norms*iter),2);
z =  1.0 - (2.0 .* uv(:,1));
r = sqrt(1- z.*z);
phi = (2 * pi) .* uv(:,2);
random_norm_2 = [r .* cos(phi), r .* sin(phi), z];

node_4_directions = random_norm_2(43:56,:);
node_5_single_dir = random_norm_2(68,:);
node_7_single_dir = random_norm_2(102,:);

random_directions([43, 45, 47, 49, 50, 54, 55, 56],:) = node_4_directions([3:7, 10:11, 14],:);
random_directions(67,:) = node_5_single_dir;
random_directions(91,:) = node_7_single_dir;
% --------------------------------------------------------------------------------------------
for i=1:len_norms
    start_idx = 1 + ((iter+1) * (i-1));
    end_idx = (iter+1) + ((iter+1) * (i-1));
    
    start_idx_rand = 1 + ((iter) * (i-1));
    end_idx_rand = (iter) + ((iter) * (i-1));
    
    xyz_directions(start_idx,1) = force_X_norm(i); % surface norm x
    xyz_directions(start_idx,2) = force_Y_norm(i); % surface norm y
    xyz_directions(start_idx,3) = force_Z_norm(i); % surface norm z
    xyz_directions(start_idx+1:end_idx,:) = random_directions(start_idx_rand:end_idx_rand,:); %random norms

    xyz_directions(start_idx+1:end_idx,1) = abs(xyz_directions(start_idx+1:end_idx,1)) * sign(force_X_norm(i));
    xyz_directions(start_idx+1:end_idx,2) = abs(xyz_directions(start_idx+1:end_idx,2)) * sign(force_Y_norm(i));
    xyz_directions(start_idx+1:end_idx,3) = abs(xyz_directions(start_idx+1:end_idx,3)) * sign(force_Z_norm(i));
    
end
% --------------------------------------------------
minimum_x_dir = min(xyz_directions(:,1));
minimum_y_dir = min(xyz_directions(:,2));
minimum_z_dir = min(xyz_directions(:,3));
maximum_x_dir = max(xyz_directions(:,1));
maximum_y_dir = max(xyz_directions(:,2));
maximum_z_dir = max(xyz_directions(:,3));
% --------------------------------------------------
% figure;
% h1 = histogram(xyz_directions(:,1),[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
% figure;
% h2 = histogram(xyz_directions(:,2),[-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0]);
% figure;
% h3 = histogram(xyz_directions(:,3),[-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3]);
% --------------------------------------------------
forces_direction_a = xyz_directions(1:15,:);
forces_direction_b = xyz_directions(16:30,:);
forces_direction_c = xyz_directions(31:45,:);
forces_direction_d = xyz_directions(46:60,:);
forces_direction_e = xyz_directions(61:75,:);
forces_direction_f = xyz_directions(76:90,:);
forces_direction_g = xyz_directions(91:105,:);
forces_direction_h = xyz_directions(106:120,:);
forces_direction_i = xyz_directions(121:135,:);
forces_direction_j = xyz_directions(136:150,:);
forces_direction_k = xyz_directions(151:165,:);
% --------------------------------------------------
%% ***SAVE #4 The direction of forces 
if save_var == 1
    save(strcat(input_folder_name,'/force_dir.mat'),'xyz_directions');
    csvwrite(strcat(input_folder_name,'/force_dir_a.csv'), forces_direction_a);
    csvwrite(strcat(input_folder_name,'/force_dir_b.csv'), forces_direction_b);
    csvwrite(strcat(input_folder_name,'/force_dir_c.csv'), forces_direction_c);
    csvwrite(strcat(input_folder_name,'/force_dir_d.csv'), forces_direction_d);
    csvwrite(strcat(input_folder_name,'/force_dir_e.csv'), forces_direction_e);
    csvwrite(strcat(input_folder_name,'/force_dir_f.csv'), forces_direction_f);
    csvwrite(strcat(input_folder_name,'/force_dir_g.csv'), forces_direction_g);
    csvwrite(strcat(input_folder_name,'/force_dir_h.csv'), forces_direction_h);
    csvwrite(strcat(input_folder_name,'/force_dir_i.csv'), forces_direction_i);
    csvwrite(strcat(input_folder_name,'/force_dir_j.csv'), forces_direction_j);
    csvwrite(strcat(input_folder_name,'/force_dir_k.csv'), forces_direction_k);
end
% --------------------------------------------------
xyz_directions = f_magnitude * xyz_directions;
% -------------------------------------------------------------------------
%% ###########################################################################################
%% Automated Data Generation: 
%  Generates simulations for p_nodes x n_directions
%  Node: the output that is saved is only displacement per simulation
%  The results for energy and other graphical simulations are not saved.
%  dir_num_rows: iter + 1 
% *** NOTE *** To regenerate the data set, if error termination occured for 
% any of the simulations, simply re-run the code, and everything will work. 
% Simulations have been tested a couple of times and all solutions will 
% converge. 
% -------------------------------------------------- 
for i=1:len_norms
    %% Specificying prescribed load node:  
    bcPrescribe_node = bcPrescribeList(i,1); % one node only. in total we have 11 nodes
    % -----------------------------------------------------------------
    for j=1:(iter+1)
        %% Specifying direction:
        % Note: magnitude is 1N, and the components of force x direction
        % i.e. the x, y and z component have this property: 
        % x^2 + y^2 + z^2 = magnitude N
        current_direction_idx = (i-1)*(iter+1) + j;
        f_mag_dir = xyz_directions(current_direction_idx, :); 
        force_X = f_mag_dir(1,1);
        force_Y = f_mag_dir(1,2);
        force_Z = f_mag_dir(1,3); 
        % -----------------------------------------------------------------
        %% Defining the FEBio input structure
        % See also |febioStructTemplate| and |febioStruct2xml| and the FEBio user
        % manual.
        % ----------------------------------------------------------------- 
        % Get a template with default settings 
        [febio_spec]=febioStructTemplate;
        % -----------------------------------------------------------------
        % febio_spec version 
        febio_spec.ATTR.version='2.5';
        % -----------------------------------------------------------------
        % Module section
        febio_spec.Module.ATTR.type='solid';
        % -----------------------------------------------------------------
        % Create control structure for use by all steps
        stepStruct.Control.analysis.ATTR.type='static';
        stepStruct.Control.time_steps=numTimeSteps;
        stepStruct.Control.step_size=1/numTimeSteps;
        stepStruct.Control.time_stepper.dtmin=dtmin;
        stepStruct.Control.time_stepper.dtmax=dtmax;
        stepStruct.Control.time_stepper.max_retries=max_retries;
        stepStruct.Control.time_stepper.opt_iter=opt_iter;
        stepStruct.Control.max_refs=max_refs;
        stepStruct.Control.max_ups=max_ups;
        % -------------------------------------------------------------------------
        % Add template based default settings to proposed control section
        [stepStruct.Control]=structComplete(stepStruct.Control,febio_spec.Control,1); % Complement provided with default if missing
        % -------------------------------------------------------------------------
        % Remove control field (part of template) since step specific control sections are used
        febio_spec=rmfield(febio_spec,'Control');    
        febio_spec.Step{1}.Control=stepStruct.Control;
        febio_spec.Step{1}.ATTR.id=1;
        % -------------------------------------------------------------------------
        % Material section
        % --> Brain
        febio_spec.Material.material{1}.ATTR.type='neo-Hookean';
        febio_spec.Material.material{1}.ATTR.id=1;
        febio_spec.Material.material{1}.density=density1;
        febio_spec.Material.material{1}.E=E_youngs1;
        febio_spec.Material.material{1}.v=nu1;
        % --> Tumour
        febio_spec.Material.material{2}.ATTR.type='neo-Hookean';
        febio_spec.Material.material{2}.ATTR.id=2;
        febio_spec.Material.material{2}.density=density2;
        febio_spec.Material.material{2}.E=E_youngs2;
        febio_spec.Material.material{2}.v=nu2;
        % -------------------------------------------------------------------------
        % Geometry section
        % -> Nodes
        febio_spec.Geometry.Nodes{1}.ATTR.name='nodeSet_all'; % The node set name
        febio_spec.Geometry.Nodes{1}.node.ATTR.id=(1:size(V,1))'; % The node id's
        febio_spec.Geometry.Nodes{1}.node.VAL=V; % The nodel coordinates
        % -------------------------------------------------------------------------
        % -> Elements
        brainElements= elementMaterialID==1;
        tumourElements=elementMaterialID==2;
        E1=E(brainElements,:);
        E2=E(tumourElements,:);
        E = [E1;E2];
        febio_spec.Geometry.Elements{1}.ATTR.type='tet4'; % Element type of this set
        febio_spec.Geometry.Elements{1}.ATTR.mat=1; % Material index for this set
        febio_spec.Geometry.Elements{1}.ATTR.name='Brain'; % Name of the element set
        febio_spec.Geometry.Elements{1}.elem.ATTR.id=(1:1:size(E1,1))'; % Element id's
        febio_spec.Geometry.Elements{1}.elem.VAL=E1;
        febio_spec.Geometry.Elements{2}.ATTR.type='tet4'; % Element type of this set
        febio_spec.Geometry.Elements{2}.ATTR.mat=2; % Material index for this set
        febio_spec.Geometry.Elements{2}.ATTR.name='Tumour'; % Name of the element set
        febio_spec.Geometry.Elements{2}.elem.ATTR.id=size(E1,1)+(1:1:size(E2,1))'; % Elements' id's
        febio_spec.Geometry.Elements{2}.elem.VAL=E2;
        % -------------------------------------------------------------------------
        % -> NodeSets
        febio_spec.Geometry.NodeSet{1}.ATTR.name='bcSupportList';
        febio_spec.Geometry.NodeSet{1}.node.ATTR.id=bcSupportList(:);
        febio_spec.Geometry.NodeSet{2}.ATTR.name='bcPrescribe_node';
        febio_spec.Geometry.NodeSet{2}.node.ATTR.id=bcPrescribe_node(:); % changed to one node only
        % -------------------------------------------------------------------------
        % Boundary condition section
        % -> Fix boundary conditions
        febio_spec.Boundary.fix{1}.ATTR.bc='x';
        febio_spec.Boundary.fix{1}.ATTR.node_set=febio_spec.Geometry.NodeSet{1}.ATTR.name;
        febio_spec.Boundary.fix{2}.ATTR.bc='y';
        febio_spec.Boundary.fix{2}.ATTR.node_set=febio_spec.Geometry.NodeSet{1}.ATTR.name;
        febio_spec.Boundary.fix{3}.ATTR.bc='z';
        febio_spec.Boundary.fix{3}.ATTR.node_set=febio_spec.Geometry.NodeSet{1}.ATTR.name;
        % -------------------------------------------------------------------------
        febio_spec.MeshData.NodeData{1}.ATTR.name='force_X';
        febio_spec.MeshData.NodeData{1}.ATTR.node_set=febio_spec.Geometry.NodeSet{2}.ATTR.name;
        febio_spec.MeshData.NodeData{1}.node.VAL=force_X;
        febio_spec.MeshData.NodeData{1}.node.ATTR.lid=(1:1:numel(bcPrescribe_node))';
        % -------------------------------------------------------------------------
        febio_spec.MeshData.NodeData{2}.ATTR.name='force_Y';
        febio_spec.MeshData.NodeData{2}.ATTR.node_set=febio_spec.Geometry.NodeSet{2}.ATTR.name;
        febio_spec.MeshData.NodeData{2}.node.VAL=force_Y;
        febio_spec.MeshData.NodeData{2}.node.ATTR.lid=(1:1:numel(bcPrescribe_node))';
        % -------------------------------------------------------------------------
        febio_spec.MeshData.NodeData{3}.ATTR.name='force_Z';
        febio_spec.MeshData.NodeData{3}.ATTR.node_set=febio_spec.Geometry.NodeSet{2}.ATTR.name;
        febio_spec.MeshData.NodeData{3}.node.VAL=force_Z;
        febio_spec.MeshData.NodeData{3}.node.ATTR.lid=(1:1:numel(bcPrescribe_node))';
        % -------------------------------------------------------------------------
        % Loads section
        % -> Prescribed nodal forces
        febio_spec.Loads.nodal_load{1}.ATTR.bc='x';
        febio_spec.Loads.nodal_load{1}.ATTR.node_set=febio_spec.Geometry.NodeSet{2}.ATTR.name;
        febio_spec.Loads.nodal_load{1}.scale.ATTR.lc=1;
        febio_spec.Loads.nodal_load{1}.scale.VAL=1;
        febio_spec.Loads.nodal_load{1}.value.ATTR.node_data=febio_spec.MeshData.NodeData{1}.ATTR.name;
        % -------------------------------------------------------------------------
        febio_spec.Loads.nodal_load{2}.ATTR.bc='y';
        febio_spec.Loads.nodal_load{2}.ATTR.node_set=febio_spec.Geometry.NodeSet{2}.ATTR.name;
        febio_spec.Loads.nodal_load{2}.scale.ATTR.lc=1;
        febio_spec.Loads.nodal_load{2}.scale.VAL=1;
        febio_spec.Loads.nodal_load{2}.value.ATTR.node_data=febio_spec.MeshData.NodeData{2}.ATTR.name;
        % -------------------------------------------------------------------------
        febio_spec.Loads.nodal_load{3}.ATTR.bc='z';
        febio_spec.Loads.nodal_load{3}.ATTR.node_set=febio_spec.Geometry.NodeSet{2}.ATTR.name;
        febio_spec.Loads.nodal_load{3}.scale.ATTR.lc=1;
        febio_spec.Loads.nodal_load{3}.scale.VAL=1;
        febio_spec.Loads.nodal_load{3}.value.ATTR.node_data=febio_spec.MeshData.NodeData{3}.ATTR.name;
        % -------------------------------------------------------------------------
        %% Defining the FEBio ouput structure
        % -> log file
        febio_spec.Output.logfile.ATTR.file=febioLogFileName;
        febio_spec.Output.logfile.node_data{1}.ATTR.file=febioLogFileName_disp;
        febio_spec.Output.logfile.node_data{1}.ATTR.data='ux;uy;uz';
        febio_spec.Output.logfile.node_data{1}.ATTR.delim=',';
        febio_spec.Output.logfile.node_data{1}.VAL=1:size(V,1);
        % -------------------------------------------------------------------------
        febio_spec.Output.logfile.node_data{2}.ATTR.file=febioLogFileName_force;
        febio_spec.Output.logfile.node_data{2}.ATTR.data='Rx;Ry;Rz';
        febio_spec.Output.logfile.node_data{2}.ATTR.delim=',';
        febio_spec.Output.logfile.node_data{2}.VAL=1:size(V,1);
        % -------------------------------------------------------------------------
        febio_spec.Output.logfile.element_data{2}.ATTR.file=febioLogFileName_strain;
        febio_spec.Output.logfile.element_data{2}.ATTR.data='E1;E2;E3';
        febio_spec.Output.logfile.element_data{2}.ATTR.delim=',';
        febio_spec.Output.logfile.element_data{2}.VAL=size(E,1);
        % -------------------------------------------------------------------------
        febio_spec.Output.logfile.element_data{1}.ATTR.file=febioLogFileName_stress;
        febio_spec.Output.logfile.element_data{1}.ATTR.data='s1;s2;s3';
        febio_spec.Output.logfile.element_data{1}.ATTR.delim=',';
        febio_spec.Output.logfile.element_data{1}.VAL=1:1:size(E,1);
        % -------------------------------------------------------------------------
        febio_spec.Output.logfile.element_data{2}.ATTR.file=febioLogFileName_strainEnergy;
        febio_spec.Output.logfile.element_data{2}.ATTR.data='sed';
        febio_spec.Output.logfile.element_data{2}.ATTR.delim=',';
        febio_spec.Output.logfile.element_data{2}.VAL=1:1:size(E,1);
        % -------------------------------------------------------------------------
        % Specifying filename for displacement (what we need)
        filename_displacement = strcat(output_folder_name,'/mat/displacement_p_node_',string(i),'_dir_',string(j),'.mat');
        filename_displacement_csv = strcat(output_folder_name,'/csv/displacement_p_node_',string(i),'_dir_',string(j),'.csv');
        % -------------------------------------------------------------------------
        %% Quick viewing of the FEBio input file structure
        
        % The |febView| function can be used to view the xml structure in a MATLAB
        % figure window.
        
        if view_specs == 1
            febView(febio_spec); %Viewing the febio file
        end
        
        % Exporting the FEBio input file
        % Exporting the febio_spec structure to an FEBio input file is done using
        % the |febioStruct2xml| function.
        
        febioStruct2xml(febio_spec,febioFebFileName); %Exporting to file and domNode
        
        %% Running the FEBio analysis
        
        % To run the analysis defined by the created FEBio input file the
        % |runMonitorFEBio| function is used. The input for this function is a
        % structure defining job settings e.g. the FEBio input file name. The
        % optional output runFlag informs the user if the analysis was run
        % succesfully.
        
        febioAnalysis.run_filename=febioFebFileName; % The input file name
        febioAnalysis.run_logname=febioLogFileName; % The name for the log file
        febioAnalysis.disp_on=1; % Display information on the command window
        febioAnalysis.disp_log_on=1; % Display convergence information in the command window
        febioAnalysis.runMode='internal';%'internal' or 'external';
        febioAnalysis.t_check=0.25; % Time for checking log file (dont set too small)
        febioAnalysis.maxtpi=1e99; % Max analysis time
        febioAnalysis.maxLogCheckTime=10; % Max log file checking time
        
        [runFlag]=runMonitorFEBio(febioAnalysis); % START FEBio NOW!!!!!!!!
        
        %% Import FEBio Results displacement
        
        if runFlag==1 % i.e. a succesful run
            
            % Importing nodal displacements from a log file
            [time_mat, N_disp_mat,~]=importFEBio_logfile(fullfile(savePath,febioLogFileName_disp)); %Nodal displacements
            time_mat=[0; time_mat(:)]; % Time
            
            N_disp_mat=N_disp_mat(:,2:end,:);
            sizImport=size(N_disp_mat);
            sizImport(3)=sizImport(3)+1;
            N_disp_mat_n=zeros(sizImport);
            N_disp_mat_n(:,:,2:end)=N_disp_mat;
            N_disp_mat=N_disp_mat_n;
            % -------------------------------------------------
            % ***SAVE #5 outputted Displacement
            if save_var == 1
                save(filename_displacement,'N_disp_mat');
                csvwrite(filename_displacement_csv,N_disp_mat);
            end
            % -------------------------------------------------
            DN=N_disp_mat(:,:,end);
            DN_magnitude=sqrt(sum(DN(:,3).^2,2));
            V_def=V+DN;
            [CF]=vertexToFaceMeasure(Fb,DN_magnitude);
            %% ###############################################################################
            %% ANIMATION DISPLACEMENT
            if animation_displacement == 1
                % Create basic view and store graphics handle to initiate animation
                hf=cFigure; % Open figure
                gtitle([febioFebFileNamePart,': Press play to animate']);
                hp=gpatch(Fb,V_def,CF,'k',1); % Add graphics object to animate
                hp.Marker='.';
                hp.MarkerSize=markerSize2;
                
                gpatch(Fb,V,0.5*ones(1,3),'none',0.25); % A static graphics object
                
                axisGeom(gca,fontSize);
                colormap(gjet(250)); colorbar;
                caxis([0 max(DN_magnitude)]);
                axis([min(V_def(:,1)) max(V_def(:,1)) min(V_def(:,2)) max(V_def(:,2)) min(V_def(:,3)) max(V_def(:,3))]); % Set axis limits statically
                camlight headlight;
                
                % Set up animation features
                animStruct.Time=time_mat; % The time vector
                for qt=1:1:size(N_disp_mat,3) % Loop over time increments
                    DN=N_disp_mat(:,:,qt); % Current displacement
                    DN_magnitude=sqrt(sum(DN.^2,2)); % Current displacement magnitude
                    V_def=V+DN; % Current nodal coordinates
                    [CF]=vertexToFaceMeasure(Fb,DN_magnitude); % Current color data to use
                    
                    % Set entries in animation structure
                    animStruct.Handles{qt}=[hp hp]; % Handles of objects to animate
                    animStruct.Props{qt}={'Vertices','CData'}; % Properties of objects to animate
                    animStruct.Set{qt}={V_def,CF}; % Property values for to set in order to animate
                end
                anim8(hf,animStruct); % Initiate animation feature
                drawnow;
            end
            %% ###############################################################################
            %% ANIMATION Energy
            if animation_energy ==1
                % Importing element strain energies from a log file
                V_DEF=N_disp_mat+repmat(V,[1 1 size(N_disp_mat,3)]);
                
                [~,E_energy,~]=importFEBio_logfile(fullfile(savePath,febioLogFileName_strainEnergy)); %Element strain energy
                
                % Remove nodal index column
                E_energy=E_energy(:,2:end,:);
                
                % Add initial state i.e. zero energy
                sizImport=size(E_energy);
                sizImport(3)=sizImport(3)+1;
                E_energy_mat_n=zeros(sizImport);
                E_energy_mat_n(:,:,2:end)=E_energy;
                E_energy=E_energy_mat_n;
                
                [FE_face,C_energy_face]=element2patch(E,E_energy(:,:,end),'tet4');
                [CV]=faceToVertexMeasure(FE_face,V,C_energy_face);
                [indBoundary]=tesBoundary(FE_face,V);
                Fb=FE_face(indBoundary,:);
                
                % Plotting the simulated results using anim8 to visualize and animate deformations
                axLim=[min(min(V_DEF,[],3),[],1); max(max(V_DEF,[],3),[],1)];
                
                % Create basic view and store graphics handle to initiate animation
                hf=cFigure; %Open figure
                title('Strain energy density')
                gtitle([febioFebFileNamePart,': Press play to animate']);
                hp1=gpatch(Fb,V_DEF(:,:,end),CV,'k',1); %Add graphics object to animate
                hp1.FaceColor='Interp';
                
                axisGeom(gca,fontSize);
                colormap(gjet(250)); colorbar;
                caxis([0 max(E_energy(:))/25]);
                axis(axLim(:)'); %Set axis limits statically
                camlight headlight;
                
                % Set up animation features
                animStruct.Time=time_mat; % The time vector
                for qt=1:1:size(N_disp_mat,3) % Loop over time increments
                    DN=N_disp_mat(:,:,qt); % Current displacement 
                    
                    [FE_face,C_energy_face]=element2patch(E,E_energy(:,:,qt),'tet4');
                    [CV]=faceToVertexMeasure(FE_face,V,C_energy_face);
                    
                    % Set entries in animation structure
                    animStruct.Handles{qt}=[hp1 hp1]; % Handles of objects to animate
                    animStruct.Props{qt}={'Vertices','CData'}; % Properties of objects to animate
                    animStruct.Set{qt}={V_DEF(:,:,qt),CV}; % Property values for to set in order to animate
                end
                anim8(hf,animStruct); % Initiate animation feature
                drawnow;
            end
        end
        %% ###############################################################################
        %% PLOTS
        if plot==1
            %% Visualize the brain and tumour models
            cFigure; hold on;
            % title('The Brain and Tumour');
            gpatch(F_brain,V_brain,'bw','k',0.5);
            gpatch(F_tumour,V_tumour,'rw','k',1);
            axisGeom;
            camlight headlight;
            gdrawnow;
            % Visualize merged set
            cFigure; hold on;
            % title('The Brain and Tumour Merged Surfaces');
            gpatch(F,V,C,'none',0.5);
            axisGeom;
            camlight headlight;
            colormap(gjet(250)); icolorbar;
            gdrawnow;
            % Visualize mesh
            hFig=cFigure; hold on;
            % title('Cut View of the Solid Mesh Depicting The Brain and Tumour Tissues','FontSize',fontSize);
            title('','FontSize',fontSize);
            optionStruct.hFig=hFig;
            optionStruct.faceAlpha1=0.2;
            optionStruct.faceAlpha2=1;
            % defaultOptionStruct.cutDir=-1;
            defaultOptionStruct.cutSide=-1;
            meshView(meshOutput,optionStruct);
            % axisGeom(gca,fontSize);
            camlight headlight;
            gdrawnow;
            %% Visualize boundary conditions and prescribed loads
            cFigure;
            hold on;
            gpatch(Fb,V,'w','k',0.7);
            gpatch(f,v,'b','k',0.5);
            % gpatch(f2,v2,'r','k',0.5);
            hl(1)=plotV(V(bcSupportList,:),'b.','MarkerSize',30);
            hl(2)=plotV(V(bcFree,:),'g.','markerSize',30);
            hl(3)=plotV(V(bcPrescribe_node,:),'r.','markerSize',25);
            legend(hl,{'BC support','BC free','BC force prescribe',});
            axisGeom;
            camlight headlight;
            drawnow;
            %% Visualize prescribed forces
            cFigure;
            subplot(1,3,1);hold on;
            title('F_x');
            gpatch(Fb,V,'w','none',0.5);
            % quiverVec([0 0 0],FX,100,'k');
            % scatterV(V(indicesHeadNodes,:),15)
            quiverVec(V(bcPrescribe_node,:),N(bcPrescribe_node,:),30,force_X);
            axisGeom; camlight headlight;
            colormap(gca,gjet(250)); colorbar;
            % -------------------------------------------------------------------
            subplot(1,3,2);hold on;
            title('F_y');
            gpatch(Fb,V,'w','none',0.5);
            % quiverVec([0 0 0],FY,100,'k');
            % scatterV(V(indicesHeadNodes,:),15)
            quiverVec(V(bcPrescribe_node,:),N(bcPrescribe_node,:),30,force_Y);
            axisGeom; camlight headlight;
            colormap(gca,gjet(250)); colorbar;
            % -------------------------------------------------------------------
            subplot(1,3,3);hold on;
            title('F_z');
            gpatch(Fb,V,'w','none',0.5);
            % quiverVec([0 0 0],FZ,100,'k');
            % scatterV(V(indicesHeadNodes,:),15)
            quiverVec(V(bcPrescribe_node,:),N(bcPrescribe_node,:),30,force_Z);
            axisGeom; camlight headlight;
            colormap(gca,gjet(250)); colorbar;
            drawnow;
        end      
    end
end